# Adenoid Hypertrophy Segmentation with MedSAM2 / MedSAM3: Research Summary

## 1. Project Goal

The target problem is **adenoid hypertrophy segmentation in nasopharyngoscopy/endoscopy images and videos**, with a stronger focus on **video segmentation**.

The intended segmentation targets are:

- **Adenoid tissue**
- **Nasopharynx / nasopharyngeal airway / choana region**

The broader clinical goal is to support **quantitative adenoid hypertrophy assessment**, for example by estimating an adenoid-to-nasopharynx or obstruction-related ratio from segmentation masks.

A strong research framing would be:

> Prompt-guided, temporally consistent adenoid/nasopharynx video segmentation for quantitative hypertrophy assessment in nasopharyngoscopy.

This framing is stronger than simply saying “apply MedSAM2 or MedSAM3 to adenoid segmentation.”

---

## 2. Is This a Worthwhile Research Direction?

Yes, it is worthwhile, especially because **ENT endoscopy is underrepresented** compared with gastrointestinal endoscopy and surgical endoscopy.

The clinical motivation is meaningful:

- Adenoid hypertrophy assessment is often subjective.
- Endoscopic grading depends on clinician interpretation.
- Quantitative segmentation could support more reproducible assessment.
- Video-based analysis can capture more information than a single static frame.

However, the novelty must be carefully defined. A weak framing would be:

> We use MedSAM2/MedSAM3 for adenoid segmentation.

A stronger framing would be:

> We study whether prompt-guided foundation models can produce temporally consistent and clinically meaningful adenoid/nasopharynx masks in nasopharyngoscopy videos with minimal human correction.

The key research question is not only whether the model can segment one frame, but whether it can maintain a useful mask across time despite camera motion, blur, secretion, occlusion, and ambiguous tissue boundaries.

---

## 3. Main Technical Challenge

The central challenge is **temporal medical video segmentation**.

A normal image segmentation model treats each frame independently:

```text
Frame 1 -> mask
Frame 2 -> mask
Frame 3 -> mask
```

This can cause mask flickering and inconsistent results.

For video, we want:

```text
Frame 1 -> mask -> memory
Frame 2 + memory -> mask -> updated memory
Frame 3 + memory -> mask -> updated memory
```

This is the main reason SAM2/MedSAM2 are relevant: they use memory to propagate segmentation across frames.

---

## 4. Why Pure Text Prompting May Be Risky

The project idea includes **text-prompt-based segmentation**, such as using prompts like:

- “adenoid”
- “nasopharynx”
- “nasopharyngeal airway”

This is ambitious but risky.

In endoscopy, the model may confuse:

- adenoid tissue with surrounding mucosa,
- nasopharyngeal cavity with dark shadows,
- specular highlights with tissue boundaries,
- secretion or blur with anatomical structure.

Therefore, pure text prompting should be treated as an experimental condition, not the guaranteed final solution.

A more realistic setup is likely:

```text
text prompt + first-frame click/box/mask + temporal propagation + sparse correction prompts
```

For clinical use, the system should allow doctors to correct failed frames.

---

## 5. MedSAM2 Architecture Overview

MedSAM2 is a medical adaptation of SAM2 for **3D medical images and videos**. It keeps the main SAM2-style architecture:

```text
Current frame
   ↓
Image encoder
   ↓
Current frame visual features
   ↓
Memory attention  ← previous-frame memory
   ↓
Memory-conditioned features
   ↓
Mask decoder  ← prompt embedding
   ↓
Predicted mask
   ↓
Memory encoder
   ↓
Memory bank
```

The major modules are:

1. **Image encoder**
2. **Prompt encoder**
3. **Memory attention module**
4. **Mask decoder**
5. **Memory encoder**
6. **Memory bank**

---

## 6. Image Encoder

The image encoder converts each video frame into visual features.

For nasopharyngoscopy, it transforms raw pixels into representations of:

- tissue texture,
- edges,
- folds,
- cavity shape,
- illumination patterns,
- specular highlights,
- possible anatomical boundaries.

For video:

```text
Frame 1 -> image features F1
Frame 2 -> image features F2
Frame 3 -> image features F3
```

The image encoder alone does not solve tracking. Temporal reasoning mainly comes from the memory system.

---

## 7. Prompt Encoder

The prompt encoder converts user prompts into embeddings that the model can use.

Typical SAM2/MedSAM2 prompts include:

- point prompt,
- box prompt,
- mask prompt.

For your task, useful prompts may include:

- a positive point on the adenoid,
- negative points outside the adenoid,
- a bounding box around the adenoid,
- a box around the nasopharyngeal airway,
- a first-frame adenoid mask,
- a first-frame nasopharynx mask.

Text prompting is more directly associated with SAM3/MedSAM3 than MedSAM2.

---

## 8. Mask Decoder

The mask decoder produces the final segmentation mask.

It receives:

1. current frame features,
2. prompt embeddings,
3. memory-conditioned information from previous frames.

It outputs a probability map or mask logits:

```text
pixel belongs to adenoid -> high probability
pixel does not belong to adenoid -> low probability
```

For your project, adenoid and nasopharynx should probably be handled as separate object tracks:

```text
Track A: adenoid
Track B: nasopharynx / airway region
```

This helps avoid mixing object identities.

---

## 9. Video Memory: The Most Important Part

SAM2/MedSAM2 handles videos by maintaining a **memory bank**.

The model does not segment every frame from scratch. Instead, it asks:

> Given what I previously segmented as the object, where is the same object in the current frame?

The simplified process is:

```text
Frame 1:
  user prompt -> mask prediction -> store memory

Frame 2:
  current frame + memory from Frame 1 -> mask prediction -> store memory

Frame 3:
  current frame + memory from Frames 1-2 -> mask prediction -> store memory
```

This helps the model maintain temporal consistency.

---

## 10. What Is Stored in Memory?

The memory is not just a raw mask. It stores learned representations related to:

- previous frame features,
- predicted masks,
- object identity,
- spatial location,
- appearance information.

A simplified memory item can be thought of as:

```text
Memory item = frame features + mask information + object identity + positional information
```

For adenoid segmentation, memory may implicitly encode:

- what the adenoid looked like earlier,
- where it was located,
- its rough shape,
- its relation to the surrounding nasopharyngeal cavity.

---

## 11. Memory Encoder

After the model predicts a mask on a frame, the memory encoder combines:

```text
current frame features + predicted mask
```

and converts them into a compact representation that can be saved in the memory bank.

This is useful because future frames need neural features, not just raw binary masks.

---

## 12. Memory Attention

Memory attention allows the current frame to attend to previous memory.

A simplified transformer view is:

```text
Current frame features = queries
Previous memory features = keys and values
```

The model learns which previous memory information is relevant to the current frame.

In an adenoid video, this helps when:

- the current frame is blurry,
- the camera moves slightly,
- lighting changes,
- the boundary is unclear,
- the adenoid is partially occluded.

---

## 13. Error Propagation and Drift

A major risk in video segmentation is **error propagation**.

If the model makes a wrong prediction and stores it into memory, future frames may inherit the mistake.

Example:

```text
Frame 5: wrong mask
   ↓
Wrong memory stored
   ↓
Frame 6 uses wrong memory
   ↓
Frame 7 error grows
```

This is called **drift**.

In endoscopy, drift can happen because of:

- camera motion,
- blur,
- secretion,
- specular reflection,
- tissue similarity,
- occlusion,
- rapid viewpoint change.

Therefore, a realistic system should support **interactive correction**:

```text
Frame 1: initial prompt
Frame 5: model drifts
Doctor adds correction click or mask
Frame 5 memory is corrected
Future frames improve
```

---

## 14. Practical Pipeline for Adenoid/Nasopharynx Video Segmentation

A practical pipeline could be:

```text
Input: nasopharyngoscopy video

Step 1: Select one or more clear keyframes
Step 2: ENT doctor provides prompt/mask for adenoid and nasopharynx
Step 3: MedSAM2 propagates masks across video
Step 4: Doctor corrects failed frames
Step 5: Corrected frames update memory
Step 6: Final masks are used for quantitative A/N or obstruction ratio
```

For two objects:

```text
Adenoid track:
  prompt -> mask -> memory -> propagation

Nasopharynx track:
  prompt -> mask -> memory -> propagation
```

---

## 15. Important Design Choices

### 15.1 Prompt Frame Selection

The first prompted frame should be clear:

- minimal blur,
- good visibility,
- adenoid fully visible,
- nasopharynx boundary visible,
- minimal secretion/reflection.

A bad initial prompt can lead to bad memory and poor propagation.

### 15.2 Forward vs Backward Propagation

Possible strategies:

```text
Forward: frame 1 -> frame T
Backward: frame T -> frame 1
Bidirectional: propagate from one or more keyframes in both directions
```

For medical videos, bidirectional propagation from corrected keyframes may be more reliable.

### 15.3 Correction Frequency

An important experiment is to measure performance versus the number of human corrections:

```text
Dice / IoU / temporal consistency vs number of correction prompts
```

This is clinically meaningful because it measures annotation efficiency.

### 15.4 Separate Object Memory

Adenoid and nasopharynx should probably be tracked separately.

This avoids identity confusion between tissue and cavity/airway space.

---

## 16. Common Failure Modes in Adenoid Endoscopy

Expected failure modes include:

| Failure Mode | Reason |
|---|---|
| Adenoid merges with surrounding mucosa | Similar color and texture |
| Nasopharynx boundary unstable | Airway/cavity is not a solid object |
| Specular highlights included in mask | Wet tissue reflection |
| Drift after blur | Wrong memory contaminates later frames |
| Over-segmentation of pharyngeal wall | Ambiguous boundary |
| Under-segmentation of adenoid folds | Folds may appear as separate structures |
| Text prompt failure | “Adenoid” may not be visually grounded enough |

---

## 17. Recommended Papers to Read

### 17.1 Core Foundation Models

1. **SAM 2: Segment Anything in Images and Videos**  
   Main baseline for promptable image/video segmentation.  
   Repo: https://github.com/facebookresearch/sam2

2. **MedSAM2: Segment Anything in 3D Medical Images and Videos**  
   Most directly relevant medical adaptation of SAM2.  
   Repo: https://github.com/bowang-lab/MedSAM2

3. **Medical SAM 2: Segment Medical Images as Video via SAM2**  
   Important alternative approach that treats medical image segmentation as video-style tracking.  
   Repo/paper should be checked for implementation details.

4. **SAM 3 / Segment Anything with Concepts**  
   Important for text/concept prompting in images and videos.  
   Repo: https://github.com/facebookresearch/sam3

5. **MedSAM3: Delving into Segment Anything with Medical Concepts**  
   Highly relevant for medical text/concept-guided segmentation.  
   Repo: https://github.com/Joey-S-Liu/MedSAM3

6. **Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation**  
   Useful for understanding limitations of vanilla SAM3 on medical images and the importance of geometric prompts.

### 17.2 Adenoid / Nasopharyngoscopy-Specific Papers

7. **Deep Learning-Based Quantification of Adenoid Hypertrophy and Its Correlation with Pediatric OSA**  
   Very relevant because it studies quantitative adenoid hypertrophy assessment from nasopharyngoscopy images.

8. **MIB-ANet: A Novel Multi-Scale Deep Network for Nasal Endoscopy-Based Adenoid Hypertrophy Grading**  
   Relevant adenoid-endoscopy grading paper. Mostly classification/grading, not video segmentation.

9. **Deep Learning-Based Automatic Adenoid Segmentation and a Novel Volume-Based Index for Adenoid Hypertrophy Assessment**  
   CBCT-based, not endoscopy, but useful for quantitative adenoid/nasopharynx ratio design.

10. **TSUBF-Net: Trans-Spatial UNet-like Network with Bi-direction Fusion for Segmentation of Adenoid Hypertrophy in CT**  
   CT-based supervised segmentation paper, useful for understanding adenoid segmentation challenges.

### 17.3 ENT / Nasal Endoscopy AI Background

11. **U-SEANNet: A Simple, Efficient and Applied U-Shaped Network for Diagnosis of Nasal Diseases on Nasal Endoscopic Images**  
   Useful for nasal endoscopy AI background.

12. **AnatomyNet: Deep Learning for Fast and Fully Automated Whole-volume Segmentation of Head and Neck Anatomy**  
   Useful for head-and-neck anatomy segmentation background.

### 17.4 General Medical SAM Papers

13. **Segment Anything in Medical Images / MedSAM**  
   Important background before reading MedSAM2/3.

14. **SAM-Med2D / SA-Med2D-20M**  
   Useful for large-scale 2D medical segmentation adaptation.

---

## 18. Suggested Reading Order

Recommended order:

```text
SAM2
→ MedSAM2
→ Medical SAM2
→ SAM3
→ MedSAM3
→ Medical SAM3
→ adenoid quantification paper
→ MIB-ANet
→ CT/CBCT adenoid segmentation papers
→ nasal endoscopy AI papers
```

The most critical papers for your exact idea are:

1. SAM2
2. MedSAM2
3. Medical SAM2
4. SAM3
5. MedSAM3
6. Deep learning-based adenoid hypertrophy quantification in nasopharyngoscopy
7. MIB-ANet

---

## 19. Public Datasets for Preliminary Testing

A major limitation is that there does not appear to be a clearly public, mask-level **adenoid/nasopharynx endoscopy video segmentation dataset**.

Therefore, public datasets should be used as proxy datasets before collecting hospital data.

### 19.1 PolypGen

PolypGen is a good candidate for a preliminary benchmark, but only as a proxy.

It is useful because:

- it is an endoscopy dataset,
- it contains video sequences,
- it has segmentation masks,
- it contains endoscopic artifacts such as blur, reflection, mucus, and illumination changes,
- it supports testing video propagation and prompt-based segmentation.

However, it is limited because:

- it is colonoscopy, not ENT endoscopy,
- polyps are lesion objects, not anatomical regions,
- adenoid/nasopharynx boundaries are more ambiguous,
- it cannot validate clinical A/N-ratio usefulness.

Conclusion:

> PolypGen is a strong engineering benchmark but a weak clinical surrogate for adenoid hypertrophy.

Use it for:

- debugging the pipeline,
- testing SAM2/MedSAM2 video propagation,
- comparing prompt types,
- checking temporal consistency,
- stress-testing endoscopic artifacts.

Do not use it to claim ENT clinical validity.

### 19.2 Other Useful Proxy Datasets

Other possible datasets include:

- laryngeal/vocal-fold endoscopic segmentation datasets,
- sinus surgery endoscopic image datasets,
- EDD2020 / EndoCV2020,
- ERS endoscopy dataset,
- Endoscapes,
- SA-Med2D-20M / SAM-Med2D.

These datasets are useful for pipeline development, but none fully replaces real adenoid/nasopharyngoscopy video data.

---

## 20. PolypGen SOTA / Baseline Results

PolypGen does not have one universal SOTA number because papers use different settings:

- still images vs video sequences,
- seen centers vs unseen centers,
- white-light vs NBI,
- prompt-based vs fully automatic segmentation,
- first-frame box/mask prompt vs no prompt.

Therefore, results should be reported by protocol.

A relevant video setting is from Polyp SAM 2 on **23 PolypGen video sequences**:

| Setting | Reported Result |
|---|---:|
| SAM2 with first-frame bounding box prompt | mDice 0.879, mIoU 0.785 |
| TransNetR supervised baseline | mDice 0.5168, mIoU 0.4717 |
| UACANet supervised baseline | mDice 0.4748, mIoU 0.4155 |
| U-Net++ supervised baseline | mDice 0.4772, mIoU 0.4272 |

Interpretation:

- SAM2-style prompt-based propagation can be very strong on endoscopic video.
- But it uses a first-frame bounding box prompt, so it is not fully automatic.
- These numbers are useful for engineering comparison, not for claiming adenoid performance.

For your work, useful PolypGen baselines include:

1. SAM2 / MedSAM2 with first-frame box or mask prompt,
2. Polyp SAM 2,
3. TransNetR,
4. UACANet,
5. U-Net++,
6. HarDNet-MSEG,
7. Polyp-PVT,
8. EndoCV2021 PolypGen challenge methods.

---

## 21. Recommended Experimental Plan

### Stage 1: Public Endoscopy Benchmark

Use public datasets such as PolypGen to test whether the pipeline works for endoscopic video segmentation.

Compare:

- SAM2,
- MedSAM2,
- SAM3,
- MedSAM3,
- supervised segmentation baselines if available.

Prompt settings:

- text only,
- point only,
- box only,
- first-frame mask,
- text + point,
- text + box,
- text + first-frame mask.

Metrics:

- Dice,
- IoU,
- boundary F-score,
- temporal consistency,
- mask flicker,
- number of correction prompts needed.

### Stage 2: ENT-Like Public Datasets

Use closer datasets such as laryngeal or sinus endoscopy if available.

Purpose:

- test domain shift from GI endoscopy to ENT endoscopy,
- evaluate robustness to upper-airway anatomy and tissue appearance.

### Stage 3: Hospital Adenoid Videos

Collect and annotate real nasopharyngoscopy videos.

Suggested annotation plan:

- 30–50 pilot videos,
- sparse keyframe masks,
- expert correction,
- separate labels for adenoid and nasopharynx/airway,
- strict annotation protocol.

Evaluate:

- adenoid Dice/IoU,
- nasopharynx Dice/IoU,
- A/N ratio error,
- clinical grading agreement,
- temporal consistency,
- correction burden.

---

## 22. Strong Possible Research Contribution

A strong contribution could be:

> A semi-automatic ENT video annotation and segmentation framework that combines text/concept prompts, first-frame spatial prompts, temporal propagation, and sparse expert correction for quantitative adenoid hypertrophy assessment.

This is stronger than a pure benchmark because it addresses the real clinical workflow.

---

## 23. Critical Risks

### Risk 1: Text Prompt Ambiguity

The word “adenoid” may not visually ground well in endoscopy frames.

Mitigation:

- compare text-only with spatial and hybrid prompts,
- do not assume text-only will work,
- use text as one condition, not the only condition.

### Risk 2: Boundary Ambiguity

Even clinicians may disagree on exact adenoid/nasopharynx boundaries.

Mitigation:

- create a strict annotation guideline,
- measure inter-annotator agreement,
- define whether nasopharynx means airway space, choana opening, or anatomical cavity.

### Risk 3: Error Propagation

Wrong masks can contaminate memory and affect later frames.

Mitigation:

- allow sparse correction,
- evaluate correction frequency,
- use bidirectional propagation,
- avoid storing low-confidence masks.

### Risk 4: Dataset Scarcity

Public datasets cannot prove clinical usefulness for adenoid hypertrophy.

Mitigation:

- use public datasets only for pipeline validation,
- collect hospital data for final clinical evaluation.

---

## 24. Bottom Line

This is a worthwhile research direction, but it should be framed carefully.

The best framing is:

> Prompt-guided, temporally consistent adenoid/nasopharynx video segmentation for quantitative adenoid hypertrophy assessment.

The project should not rely only on pure text prompting. A more realistic and publishable system would combine:

```text
text/concept prompt
+ spatial prompt
+ temporal memory propagation
+ sparse expert correction
+ quantitative clinical measurement
```

PolypGen is useful for preliminary testing, but the final claim must be validated on real ENT/nasopharyngoscopy videos.
