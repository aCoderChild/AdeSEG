import torch
from modeling.current_frame_refinement import CurrentFrameRefinement
from modeling.regional_refinement import RegionalRefinement, refinement_loss


def test_shared_initialization_and_no_context_parameters():
    torch.manual_seed(20261031)
    original = RegionalRefinement()
    torch.manual_seed(20261031)
    head = CurrentFrameRefinement()
    assert torch.equal(head.project.conv.weight, original.project.conv.weight)
    assert torch.equal(head.fusion.conv.weight, original.head.conv_out2.conv.weight[:,:64])
    assert torch.equal(head.refine.conv.weight, original.head.conv_out3.conv.weight)
    assert sum(p.numel() for p in head.parameters()) < sum(p.numel() for p in original.parameters())


def test_identity_context_independence_presence_and_gradients():
    head = CurrentFrameRefinement().eval()
    inputs = [torch.randn(2,256,8,8), torch.randn(2,64,3,1),
              torch.ones(2,3,dtype=torch.bool), torch.randn(2,1,32,32), torch.ones(2)]
    assert torch.equal(head(*inputs), inputs[3])
    with torch.no_grad():
        head.output.conv.weight.fill_(.01)
    first = head(*inputs)
    inputs[1] = torch.full_like(inputs[1], float('nan'))
    inputs[2] = torch.zeros_like(inputs[2])
    assert torch.equal(first, head(*inputs))
    refinement_loss(first, torch.zeros_like(first)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert head.project.conv.weight.grad.abs().sum() > 0
    inputs[4].zero_()
    assert torch.equal(head(*inputs), inputs[3])
