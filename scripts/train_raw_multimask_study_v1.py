#!/usr/bin/env python3
"""Six-arm fixed-read raw-candidate pilot. No live video; freeze before scoring."""
import argparse,json,math,random,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import save_binary_mask
from modeling.lean_channel_attention import author_sha256
from scripts.memory_token_io import digest,validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import load_target,to_video
from scripts.train_region_mixed_loss_v1 import environment_identity,pixel_loss,state_sha256
from scripts.raw_multimask_shared_v1 import ARMS,LOSSES,SEED,EPOCHS,LOSS_FORMULAS,dependency_hashes,load_manifests,entry_for,inputs,model_for

PARITY_KEYS=(('seq4',24),('seq4',21),('seq6',90),('seq13',62))

def parity(m,mhash,entries,rhash,output):
    torch.manual_seed(SEED);cpu=model_for('raw').train()
    torch.nn.init.normal_(cpu.head[-1].weight,mean=0,std=.01);torch.nn.init.normal_(cpu.raw_adapter.weight,mean=0,std=.01)
    gpu=model_for('raw').train();gpu.load_state_dict(cpu.state_dict());gpu.to('mps')
    checks=[]
    for seq,frame in PARITY_KEYS:
        item=next(x for x in m['samples'] if x['split']=='train' and x['sequence']==seq and x['frame']==frame)
        for kind in LOSSES:
            values=[]
            for model,device in ((cpu,'cpu'),(gpu,'mps')):
                model.zero_grad(set_to_none=True);args,shape=inputs(item,entry_for(entries,item),device)
                target=load_target(item,shape).to(device);out=model(*args);loss=pixel_loss(to_video(out,shape),target,kind)
                if not torch.isfinite(out).all() or not torch.isfinite(loss):raise ValueError('Nonfinite parity output')
                loss.backward();grad=[p.grad.detach().cpu().clone() for p in model.parameters()]
                if not all(torch.isfinite(x).all() for x in grad):raise ValueError('Nonfinite parity gradient')
                values.append((out.detach().cpu(),loss.detach().cpu(),grad,(to_video(out.detach(),shape)>0).cpu()))
            torch.testing.assert_close(values[0][0],values[1][0],rtol=0,atol=1e-3);torch.testing.assert_close(values[0][1],values[1][1],rtol=1e-3,atol=1e-3)
            # Pixel-level equivalence at the actual video geometry, with a
            # predeclared maximum of 20 pixels and 1e-5 of the image.
            left_mask=values[0][3];right_mask=values[1][3]
            mask_disagreement=int((left_mask!=right_mask).sum());pixel_count=left_mask.numel()
            if mask_disagreement>20 or mask_disagreement/pixel_count>1e-5:raise ValueError('CPU/MPS native-size masks differ beyond fixed tolerance')
            diff=maximum=0.0
            for a,b in zip(values[0][2],values[1][2],strict=True):
                torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-3);diff=max(diff,float((a-b).abs().max()));maximum=max(maximum,float(a.abs().max()))
            checks.append({'sequence':seq,'frame':frame,'capture_device':item['capture_device'],'loss':kind,'cpu_loss':float(values[0][1]),'mps_loss':float(values[1][1]),'maximum_output_difference':float((values[0][0]-values[1][0]).abs().max()),'maximum_cpu_output_magnitude':float(values[0][0].abs().max()),'maximum_mps_output_magnitude':float(values[1][0].abs().max()),'native_mask_disagreement':mask_disagreement,'native_pixel_count':pixel_count,'maximum_gradient_difference':diff,'maximum_gradient_magnitude':maximum,'maximum_mps_gradient_magnitude':max(float(x.abs().max()) for x in values[1][2])})
    out=validate_output(output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'manifest_sha256':mhash,'raw_manifest_sha256':rhash,'dependency_sha256':dependency_hashes(),'environment':environment_identity(),'checks':checks,'mask_tolerance':{'max_pixels':20,'max_fraction':1e-5},'logit_tolerance':{'absolute':0.001,'relative':0.0},'scope':'Training-read parity only; no validation labels or live video.'}
    (out/'parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

def validate_plan(plan_path,parity_path,mhash,rhash):
    p=json.loads(plan_path.read_text());r=json.loads(parity_path.read_text())
    expected={'manifest_sha256':mhash,'raw_manifest_sha256':rhash,'dependency_sha256':dependency_hashes(),'environment':environment_identity(),'seed':SEED,'epochs':EPOCHS,'arms':list(ARMS),'losses':list(LOSSES),'loss_formulas':LOSS_FORMULAS,'optimizer':'Adam','learning_rate':.0003,'weight_decay':.0001,'parameter_count':48453,'evaluation':'freeze all six arms before independent label scoring'}
    if any(p.get(k)!=v for k,v in expected.items()) or p.get('parity_sha256')!=digest(parity_path):raise ValueError('Plan differs')
    if not r.get('passed') or r.get('mask_tolerance')!={'max_pixels':20,'max_fraction':1e-5} or r.get('logit_tolerance')!={'absolute':0.001,'relative':0.0} or any(r.get(k)!=expected[k] for k in ('manifest_sha256','raw_manifest_sha256','dependency_sha256','environment')):raise ValueError('Parity differs')
    checks=r.get('checks',[]);keys={(x['sequence'],x['frame'],x['loss']) for x in checks}
    if len(checks)!=8 or keys!={(s,f,k) for s,f in PARITY_KEYS for k in LOSSES} or any(any(not math.isfinite(x[k]) for k in ('cpu_loss','mps_loss','maximum_output_difference','maximum_cpu_output_magnitude','maximum_mps_output_magnitude','maximum_gradient_difference','maximum_gradient_magnitude','maximum_mps_gradient_magnitude')) or x['maximum_output_difference']>0.001 or x['native_mask_disagreement']>20 or x['native_mask_disagreement']/x['native_pixel_count']>1e-5 for x in checks):raise ValueError('Parity coverage differs')
    return p

def train(m,mhash,entries,rhash,arm,kind,out,plan_path,parity_path):
    validate_plan(plan_path,parity_path,mhash,rhash)
    trainrows=sorted((x for x in m['samples'] if x['split']=='train'),key=lambda x:(x['sequence'],x['frame']))
    valrows=sorted((x for x in m['samples'] if x['split']=='validation'),key=lambda x:(x['sequence'],x['frame']))
    for x in m['samples']:
        if digest(x['image'])!=x['image_sha256']:raise ValueError('Image changed')
    for x in trainrows:
        if digest(x['label'])!=x['label_sha256']:raise ValueError('Training label changed')
    out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(SEED);torch.mps.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    model=model_for(arm).to('mps').train();initial=state_sha256(model);opt=torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=.0001);history=[]
    for epoch in range(EPOCHS):
        samples=[]
        for idx in torch.randperm(len(trainrows)).tolist():
            x=trainrows[idx];args,shape=inputs(x,entry_for(entries,x),'mps');target=load_target(x,shape).to('mps')
            opt.zero_grad(set_to_none=True);prediction=model(*args);loss=pixel_loss(to_video(prediction,shape),target,kind)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradient')
            grad=max(float(p.grad.detach().abs().max().cpu()) for p in model.parameters() if p.grad is not None);opt.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite parameter')
            samples.append({'sequence':x['sequence'],'frame':x['frame'],'capture_device':x['capture_device'],'loss':float(loss.detach().cpu()),'maximum_gradient':grad})
        history.append({'epoch':epoch+1,'mean_loss':float(np.mean([x['loss'] for x in samples])),'samples':samples})
        (out/'progress.json').write_text(json.dumps({'history':history},indent=2)+'\n');print(f'{arm}/{kind} epoch {epoch+1}/{EPOCHS}',flush=True)
    payload={'format':'raw_multimask_refiner_v1','research_only':True,'training_steps':len(trainrows)*EPOCHS,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'backbone_sha256':m['checkpoint_sha256'],'feature_pipeline_sha256':m['feature_pipeline_sha256'],'train_manifest_sha256':mhash,'raw_manifest_sha256':rhash,'author_sha256':author_sha256(),'raw_model_sha256':digest(ROOT/'modeling/raw_multimask_refiner.py'),'conditioned_source_sha256':digest(ROOT/'modeling/foreground_conditioned_refiner.py'),'base_source_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),'raw_mode':arm,'gate_mode':'region','contrast_head':True,'loss_kind':kind,'loss_formula':LOSS_FORMULAS[kind],'seed':SEED,'epochs':EPOCHS,'initial_state_sha256':initial,'parameter_count':sum(p.numel() for p in model.parameters()),'dependency_sha256':dependency_hashes(),'environment':environment_identity(),'plan_sha256':digest(plan_path),'parity_sha256':digest(parity_path)}
    ck=out/'checkpoint.pt';torch.save(payload,ck)
    fitted=model_for(arm);fitted.load_trained_state(torch.load(ck,map_location='cpu',weights_only=True),backbone_sha256=m['checkpoint_sha256'],feature_pipeline_sha256=m['feature_pipeline_sha256'],train_manifest_sha256=mhash,raw_manifest_sha256=rhash);fitted.to('mps').eval()
    pred=out/'frozen_predictions';pred.mkdir();rows=[]
    targets=valrows+[next(x for x in trainrows if x['sequence']=='seq4' and x['frame']==i) for i in (21,24)]
    with torch.no_grad():
        for x in targets:
            args,shape=inputs(x,entry_for(entries,x),'mps');values={'baseline':(to_video(args[2],shape)>0).cpu().numpy()[0,0],'refined':(to_video(fitted(*args),shape)>0).cpu().numpy()[0,0]}
            row={'sequence':x['sequence'],'frame':x['frame'],'split':x['split'],'shape':list(shape)}
            for name,mask in values.items():
                path=pred/f"{x['sequence']}_{x['frame']:06d}_{name}.png";save_binary_mask(mask,path);row[name]=str(path);row[name+'_sha256']=digest(path)
            rows.append(row)
    frozen=out/'frozen_predictions.json';frozen.write_text(json.dumps({'checkpoint_sha256':digest(ck),'rows':rows},indent=2)+'\n')
    complete={'passed':True,'arm':arm,'loss_kind':kind,'seed':SEED,'epochs':EPOCHS,'manifest_sha256':mhash,'raw_manifest_sha256':rhash,'plan_sha256':digest(plan_path),'checkpoint_sha256':digest(ck),'frozen_predictions_sha256':digest(frozen),'initial_state_sha256':initial,'history':history,'validation_labels_opened':False}
    (out/'completion.json').write_text(json.dumps(complete,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('data-preflight','raw-audit','output'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--mode',choices=('parity','train'),required=True);p.add_argument('--arm',choices=ARMS);p.add_argument('--loss',choices=LOSSES);p.add_argument('--plan',type=Path);p.add_argument('--parity-receipt',type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);environment_identity()
    m,mhash,entries,rhash=load_manifests(a.data_preflight,a.raw_audit)
    if a.mode=='parity':parity(m,mhash,entries,rhash,a.output)
    else:
        if None in (a.arm,a.loss,a.plan,a.parity_receipt):raise ValueError('Training requires arm, loss, plan and parity')
        train(m,mhash,entries,rhash,a.arm,a.loss,a.output,a.plan,a.parity_receipt)
if __name__=='__main__':main()
