# README

- Document ID: 1krSaKeQygxDfqtY4wXM3gcIMrxHXYwfPXDor_U529BE
- Revision ID: ANLCKQkcVloV89g6hdqQL5bJCr8BXdWR33mQnGvqpZbbp193V30Af-6kIlQ_MPiM9zl4SNLNd9-gfVHLr-WMHLBnMKlv3vC5lPvE31Hm2gJJ
- Selected tab: all
- Protected controls: 0
- Opaque controls: 0
- Authoritative dropdowns: 0

Protected-control annotations are preservation instructions. Do not insert their displayed placeholder text to recreate a native control.

## SAM modifications in past papers (t.9tweai1haq5r)

[P00001 | 1:194 | NORMAL_TEXT]
Here's a synthesis of the literature on how SAM2 has been modified for endoscopy (and closely related surgical) video segmentation. The work clusters into a few distinct types of modification.

[P00002 | 194:240 | NORMAL_TEXT]
1. Memory bank and memory-attention redesigns

[P00003 | 240:497 | NORMAL_TEXT]
This is by far the most active area, because SAM2's default memory design — which stores recent frames in temporal order and selects them greedily — copes poorly with the rapid instrument motion, occlusion, and frame redundancy typical of endoscopic video.

[P00004 | 497:897 | NORMAL_TEXT | LIST id=kix.udbklw56l1la level=0]
MA-SAM2 (Memory-Augmented SAM2) replaces the sequential memory bank with two new modules: a context-aware memory and an occlusion-resilient memory, so tracking survives instrument overlap and temporary disappearance. It's training-free and uses a single mask prompt per instrument category at first appearance, reporting gains of 4.36% and 6.1% over SAM2 on the EndoVis2017 and EndoVis2018 datasets.

[P00005 | 897:1149 | NORMAL_TEXT | LIST id=kix.udbklw56l1la level=0]
SurgSAM-2 (Surgical SAM 2) adds an Efficient Frame Pruning mechanism that dynamically manages the memory bank by selectively retaining only the most informative frames, reducing memory usage and computational cost, aimed at real-time use.[https://www.researchgate.net/publication/398851429_Memory-Enhanced_SAM3_for_Occlusion-Robust_Surgical_Instrument_Segmentation](https://www.researchgate.net/publication/398851429_Memory-Enhanced_SAM3_for_Occlusion-Robust_Surgical_Instrument_Segmentation)[ResearchGate](https://www.researchgate.net/publication/398851429_Memory-Enhanced_SAM3_for_Occlusion-Robust_Surgical_Instrument_Segmentation)

[P00006 | 1149:1351 | NORMAL_TEXT | LIST id=kix.udbklw56l1la level=0]
TSMS-SAM2 partitions stored frame features into short-term and long-term memory and applies a memory splitting-and-pruning filter, moving away from over-reliance on similarity to the most recent frame.

[P00007 | 1351:1712 | NORMAL_TEXT | LIST id=kix.udbklw56l1la level=0]
FreeVPS (video polyp segmentation) keeps SAM2 training-free but adds an inter-association refinement module that adaptively updates the memory bank to prevent error propagation over time, combined with an intra-association filtering step to cut false positives — specifically to counter the "snowball" error accumulation during long colonoscopy tracking.[https://arxiv.org/pdf/2508.19705](https://arxiv.org/pdf/2508.19705)[arxiv](https://arxiv.org/pdf/2508.19705)

[P00008 | 1712:1933 | NORMAL_TEXT | LIST id=kix.udbklw56l1la level=0]
MedSAM-2 (Zhu et al.) introduces a Self-Sorting Memory Bank that selects the most confident embeddings based on the confidence predictions from the mask decoder, which also unlocks a "one-prompt" segmentation mode.[https://arxiv.org/pdf/2408.00874](https://arxiv.org/pdf/2408.00874)[arxiv](https://arxiv.org/pdf/2408.00874)

[P00009 | 1933:2111 | NORMAL_TEXT | LIST id=kix.udbklw56l1la level=0]
SAMed-2 uses a confidence-driven memory bank with selective update/retrieval plus temporal adapters in each transformer block to strengthen inter-slice/inter-frame correlations.

[P00010 | 2111:2171 | NORMAL_TEXT]
2. Adapter-based and parameter-efficient fine-tuning (PEFT)

[P00011 | 2171:2293 | NORMAL_TEXT]
Rather than retraining the whole model, these insert small trainable modules to bridge the natural-to-medical domain gap.

[P00012 | 2293:2505 | NORMAL_TEXT | LIST id=kix.ylpj4wjpvwr8 level=0]
SAM2-Adapter inserts bottleneck adapter modules (following the earlier SAM-Adapter/Med-SA design of two MLPs around a nonlinearity) into the encoder and decoder for downstream tasks including polyp segmentation.

[P00013 | 2505:2584 | NORMAL_TEXT | LIST id=kix.ylpj4wjpvwr8 level=0]
LoRA-based fine-tuning (Yu et al.) applies low-rank adapters for surgical use.

[P00014 | 2584:2743 | NORMAL_TEXT | LIST id=kix.ylpj4wjpvwr8 level=0]
The channel-attention polyp model introduces a learnable prompt layer within the Transformer blocks while keeping most of the pretrained encoder frozen.[https://ojs.sgsci.org/journals/amr/article/view/311](https://ojs.sgsci.org/journals/amr/article/view/311)[Sgsci](https://ojs.sgsci.org/journals/amr/article/view/311)

[P00015 | 2743:2768 | NORMAL_TEXT]
3. Decoder modifications

[P00016 | 2768:3056 | NORMAL_TEXT | LIST id=kix.cs7alanh882v level=0]
A generalizable polyp model pairs the fine-tuned SAM2 encoder with a full-scale skip connection structure as a decoder to integrate multi-scale semantic features, plus a channel-attention-enhanced decoder, and reports improvements over prior methods on Kvasir-SEG and CVC-ClinicDB.[https://ojs.sgsci.org/journals/amr/article/view/311](https://ojs.sgsci.org/journals/amr/article/view/311)[Sgsci](https://ojs.sgsci.org/journals/amr/article/view/311)

[P00017 | 3056:3220 | NORMAL_TEXT | LIST id=kix.cs7alanh882v level=0]
SurgiSAM2 fine-tunes specifically the image encoder and mask decoder on anatomical tissue, achieving a 17.9% relative WMDC gain compared to the baseline SAM 2.[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12528661/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12528661/)[nih](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12528661/)

[P00018 | 3220:3250 | NORMAL_TEXT]
4. Prompting-strategy changes

[P00019 | 3250:3449 | NORMAL_TEXT | LIST id=kix.bp23dk1cvzz4 level=0]
SurgicalSAM introduces class-level prototype prompts, and SP-SAM focuses on part-level prompting, both replacing manual point/box prompts with learned or structured prompts for surgical instruments.

[P00020 | 3449:3554 | NORMAL_TEXT | LIST id=kix.bp23dk1cvzz4 level=0]
Several models (MA-SAM2, MedSAM-2) shift to "one-prompt" schemes to minimize per-frame user interaction.

[P00021 | 3554:3611 | NORMAL_TEXT]
5. Full fine-tuning with large-scale medical pretraining

[P00022 | 3611:3938 | NORMAL_TEXT | LIST id=kix.y4xc0r70pr2a level=0]
MedSAM2 (Ma et al.) is a transfer-learning pipeline fine-tuned on a large corpus that explicitly includes endoscopy — over 455,000 3D image-mask pairs spanning CT, PET, and MRI, plus 76,000 video frames from ultrasound (19,232) and endoscopy (56,462) — while retaining the Hiera backbone and memory-attention module.[https://neurohive.io/en/state-of-the-art/medsam2-open-source-sota-3d-medical-image-and-video-segmentation-model/](https://neurohive.io/en/state-of-the-art/medsam2-open-source-sota-3d-medical-image-and-video-segmentation-model/)[Neurohive](https://neurohive.io/en/state-of-the-art/medsam2-open-source-sota-3d-medical-image-and-video-segmentation-model/)

[P00023 | 3938:4079 | NORMAL_TEXT | LIST id=kix.y4xc0r70pr2a level=0]
BioSAM-2 / Biomedical SAM 2 uses large biomedical pretraining to improve generalization and reportedly matches or exceeds specialist models.

[P00024 | 4079:4113 | NORMAL_TEXT]
6. Data and temporal augmentation

[P00025 | 4113:4338 | NORMAL_TEXT | LIST id=kix.nrt7q8dd2fv8 level=0]
TSMS-SAM2 also contributes a multi-temporal-scale video sampling augmentation that samples clips at multiple frame rates to improve robustness against rapid object motion and to offset the scarcity of labeled surgical video.

[P00026 | 4338:5013 | NORMAL_TEXT]
A couple of framing notes that recur across the surveys: the underlying motivation is consistent — the original SAM2 was trained on natural images and videos characterized by strong edge information, which differs significantly from medical images that often exhibit low contrast and weak boundaries, so zero-shot performance varies a lot by dataset. And one useful taxonomy from a 2025 ultrasound-focused study divides SAM2 medical adaptations into roughly four overlapping directions: modality adaptation through fine-tuning or training-free strategies, memory redesign, adapter/PEFT insertion, and prompting changes — which maps well onto the categories above.[https://arxiv.org/html/2408.12889](https://arxiv.org/html/2408.12889)[arXiv](https://arxiv.org/html/2408.12889)[arxiv](https://arxiv.org/pdf/2511.05731)

[P00027 | 5013:5448 | NORMAL_TEXT]
A few caveats for your review: much of this work is evaluated on surgical-instrument datasets (EndoVis2017/2018, EndoNeRF, SurgToolLoc) rather than diagnostic endoscopy per se, and colonoscopy polyp segmentation forms a somewhat separate sub-thread. Many of the "endoscopy" claims for general medical models (MedSAM2, BioSAM-2) come from endoscopy being one modality within a multi-modal training set, not a dedicated endoscopy model.

## Outdated (t.tqp7j8xdnxrr)

[P00028 | 1:27 | NORMAL_TEXT]
STATUS NOTE — 19 AUG 2026

[P00029 | 27:566 | NORMAL_TEXT]
The reliability-aware memory material below is retained as prior-work analysis and as a baseline history. It is no longer the main proposed method. The current proposal is Temporal Proxy Steering: learn a temporal adaptation residual with a lightweight P0/PT proxy pair and apply that residual to a frozen video segmentation foundation model, while keeping one external dynamic probability state. Use this tab mainly as literature evidence for temporal drift, memory robustness, probability fusion, uncertainty and lightweight adaptation.

[P00030 | 566:567 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00031 | 567:749 | NORMAL_TEXT]
Problem: Application of SAM/MedSAM with dynamic probability mask state and calibrated reliability gate for video endoscopy polyp segmentation. (BOLD are for up-to-now contributions)

[P00032 | 749:750 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00033 | 750:945 | NORMAL_TEXT]
Contribution: A lightweight, endoscopy-specific reliability-calibrated memory controller for MedSAM2 that decides whether each predicted frame should update, freeze or reset the temporal memory.

[P00034 | 945:946 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00035 | 946:951 | NORMAL_TEXT]
Cần:

[P00036 | 951:1034 | NORMAL_TEXT | LIST id=kix.rj0annza48nb level=0]
giải quyết bài toán gì (abstract), không giải quyết bài toán của mình => loại luôn

[P00037 | 1034:1091 | NORMAL_TEXT | LIST id=kix.rj0annza48nb level=0]
nó chỉ ra vấn đề của SAM là gì (abstract + introduction)

[P00038 | 1091:1153 | NORMAL_TEXT | LIST id=kix.rj0annza48nb level=0]
nó giải quyết vấn đề đấy như thế nào (abstract + methodology)

[P00039 | 1153:1310 | NORMAL_TEXT | LIST id=kix.rj0annza48nb level=0]
kết quả nó như thế nào ( dùng dataset gì + results - nhìn bảng biểu) => mình cần phải xem cái nó thêm vào có giải quyết được phần vấn đề nó đặt ra hay không

[P00040 | 1310:1482 | NORMAL_TEXT | LIST id=kix.rj0annza48nb level=0]
Nếu không có / không nhìn thấy trong bảng (kiểu chỉ là kết quả sau cùng end-to-end, không có ablations so sánh kiến trúc không có mođun đấy với cái có) => limitations ở đó

[P00041 | 1482:1483 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00042 | 1483:1581 | NORMAL_TEXT | LIST id=kix.ykgo0yjxui7t level=0]
sau khi xong các bài => so sánh => nếu tăng so với baseline thì có thể cái gì chung khiến nó tăng

[P00043 | 1581:1582 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00044 | 1582:1660 | HEADING_3]
1. [Medical SAM 2: Segment Medical Images as Video via SAM2 — Zhu et al., 2024](https://arxiv.org/pdf/2408.00874)

[P00045 | 1660:1755 | NORMAL_TEXT]
Why it is essential: This is probably the closest paper to your phrase “adaptive memory bank.”

[P00046 | 1755:1841 | NORMAL_TEXT]
The paper introduces a self-sorting memory bank that selects stored embeddings using:

[P00047 | 1841:1863 | NORMAL_TEXT | LIST id=kix.kegxakm00rmf level=0]
predicted confidence;

[P00048 | 1863:1886 | NORMAL_TEXT | LIST id=kix.kegxakm00rmf level=0]
feature dissimilarity;

[P00049 | 1886:1927 | NORMAL_TEXT | LIST id=kix.kegxakm00rmf level=0]
selection independent of temporal order.

[P00050 | 1927:2044 | NORMAL_TEXT]
Its purpose is to retain informative memories and improve one-prompt segmentation across medical images and volumes.

[P00051 | 2044:2109 | NORMAL_TEXT]
Overlap with your idea: Very high for adaptive memory selection.

[P00052 | 2109:2153 | NORMAL_TEXT]
Difference from your intended contribution:

[P00053 | 2153:2211 | NORMAL_TEXT | LIST id=kix.y631q8rcmkub level=0]
It mainly addresses universal 2D/3D medical segmentation.

[P00054 | 2211:2272 | NORMAL_TEXT | LIST id=kix.y631q8rcmkub level=0]
Its confidence is used for ranking and selecting embeddings.

[P00055 | 2272:2450 | NORMAL_TEXT | LIST id=kix.y631q8rcmkub level=0]
It does not specifically study endoscopy video corruption, propagation drift, detector-prompt disagreement, or whether the current frame should be rejected from memory entirely.

[P00056 | 2450:2531 | NORMAL_TEXT | LIST id=kix.y631q8rcmkub level=0]
It is not evaluated as a reliability-control system on PolypGen video sequences.

[P00057 | 2531:2600 | NORMAL_TEXT]
Read for: memory-bank implementation and confidence-based retrieval.

[P00058 | 2600:2602 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00059 | 2602:2657 | HEADING_3]
Kiến trúc như sam2 + Phương pháp chọn ảnh hợp lý hơn: 

[P00060 | 2657:2669 | HEADING_3 | LIST id=kix.h0sgks7m3thc level=0]
Lọc đầu vào

[P00061 | 2669:2690 | HEADING_3 | LIST id=kix.h0sgks7m3thc level=0]
Loại dần trùng lặp? 

[P00062 | 2690:2728 | HEADING_3 | LIST id=kix.h0sgks7m3thc level=0]
Đánh trọng số cao cho ảnh tương đồng 

[P00063 | 2728:2729 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00064 | 2729:2746 | HEADING_3]
Có kết quả +19% 

[P00065 | 2746:2747 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00066 | 2747:2786 | HEADING_3]
Có nền Lý thuyết giải thích cho Method

[P00067 | 2786:2787 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00068 | 2787:2788 | HEADING_3]
⟦EMPTY PARAGRAPH⟧

[P00069 | 2788:2867 | HEADING_3]
2. [MedSAM2: Segment Anything in 3D Medical Images and Videos — Ma et al., 2025](https://arxiv.org/pdf/2504.03600)

[P00070 | 2867:3016 | NORMAL_TEXT]
This is the MedSAM2 foundation model you are currently using. It is trained on large-scale medical image and video data and builds directly on SAM2.

[P00071 | 3016:3215 | NORMAL_TEXT]
Some descriptions of the model report a confidence memory bank, a calibration head and weighted memory retrieval, with more similar or confident frames receiving larger weights.[https://pmc.ncbi.nlm.nih.gov/articles/PMC11428920/?utm_source=chatgpt.com](https://pmc.ncbi.nlm.nih.gov/articles/PMC11428920/?utm_source=chatgpt.com)[PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11428920/?utm_source=chatgpt.com)

[P00072 | 3215:3256 | NORMAL_TEXT]
Important consequence for your novelty: 

[P00073 | 3256:3329 | NORMAL_TEXT]
You should not claim: “We introduce confidence-aware memory to MedSAM2.”

[P00074 | 3329:3383 | NORMAL_TEXT]
That idea is already present in related MedSAM2 work.

[P00075 | 3383:3410 | NORMAL_TEXT]
A stronger distinction is:

[P00076 | 3410:3532 | NORMAL_TEXT]
“We introduce reliability-calibrated memory admission and update control for MedSAM2 under endoscopic video degradation.”

[P00077 | 3532:3560 | NORMAL_TEXT]
The distinction is between:

[P00078 | 3560:3606 | NORMAL_TEXT | LIST id=kix.voqnuucgyn1p level=0]
selecting or weighting existing memories, and

[P00079 | 3606:3683 | NORMAL_TEXT | LIST id=kix.voqnuucgyn1p level=0]
deciding whether a new prediction is reliable enough to enter memory at all.

[P00080 | 3683:3685 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00081 | 3685:3808 | NORMAL_TEXT]
Phế, Tập trung vào finetune, down scale + prompt, Chỉ đưa ra vấn đề của 8 frame queue(SAM2), không giải pháp (future work)

[P00082 | 3808:3896 | HEADING_3]
3. [SAMed-2: Selective Memory Enhanced Medical Segment Anything Model — Yan et al., 2025](https://arxiv.org/pdf/2507.03698)

[P00083 | 3896:3910 | NORMAL_TEXT]
SAMed-2 adds:

[P00084 | 3910:3930 | NORMAL_TEXT | LIST id=kix.4gto5cey3ecv level=0]
a temporal adapter;

[P00085 | 3930:3968 | NORMAL_TEXT | LIST id=kix.4gto5cey3ecv level=0]
a confidence-driven memory mechanism;

[P00086 | 3968:4004 | NORMAL_TEXT | LIST id=kix.4gto5cey3ecv level=0]
storage of high-certainty features;

[P00087 | 4004:4032 | NORMAL_TEXT | LIST id=kix.4gto5cey3ecv level=0]
similarity-based retrieval.

[P00088 | 4032:4137 | NORMAL_TEXT]
Its motivation includes noisy medical data and avoiding unreliable information being retained in memory.

[P00089 | 4137:4152 | NORMAL_TEXT]
Overlap: High.

[P00090 | 4152:4169 | NORMAL_TEXT]
Main difference:

[P00091 | 4169:4243 | NORMAL_TEXT | LIST id=kix.aq8qlzo3hcra level=0]
It focuses on large-scale multi-task and multimodal medical segmentation.

[P00092 | 4243:4302 | NORMAL_TEXT | LIST id=kix.aq8qlzo3hcra level=0]
Its confidence mechanism is learned during model training.

[P00093 | 4302:4399 | NORMAL_TEXT | LIST id=kix.aq8qlzo3hcra level=0]
It is not specifically designed around endoscopic degradation signals or online drift detection.

[P00094 | 4399:4526 | NORMAL_TEXT | LIST id=kix.aq8qlzo3hcra level=0]
It does not study prompt reliability, correction triggering or explicit freeze/update decisions in MedSAM2-based polyp videos.

[P00095 | 4526:4591 | NORMAL_TEXT]
Read for: selective storage and confidence-driven memory design.

[P00096 | 4591:4593 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00097 | 4593:4697 | NORMAL_TEXT]
Khá tương đồng bài 1, khác 1 chút về kiến trúc +9%. Sờ vào Img Encoder (Phần giúp tăng kqua nhiều nhất)

[P00098 | 4697:4847 | HEADING_3]
4. [Unsupervised Quality Control and Enhancement of Polyp Segmentation in Colonoscopy Videos Using Spatiotemporal Consistency — Li et al., MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/2132_paper.pdf)

[P00099 | 4847:4904 | NORMAL_TEXT]
This is one of the closest papers to your exact problem.

[P00100 | 4904:5274 | NORMAL_TEXT]
It uses SAM2 to propagate masks between neighboring frames and compares propagated masks with the original model predictions. The resulting agreement becomes an unsupervised Segmentation Quality Assessment score. Frames judged unreliable are re-segmented using information from higher-quality frames. It is evaluated on both SUN-SEG and PolypGen, and code is available.

[P00101 | 5274:5298 | NORMAL_TEXT]
Overlap with your idea:

[P00102 | 5298:5324 | NORMAL_TEXT | LIST id=kix.r7uwf2w9ada1 level=0]
polyp video segmentation;

[P00103 | 5324:5369 | NORMAL_TEXT | LIST id=kix.r7uwf2w9ada1 level=0]
reliability assessment without ground truth;

[P00104 | 5369:5396 | NORMAL_TEXT | LIST id=kix.r7uwf2w9ada1 level=0]
SAM2 temporal propagation;

[P00105 | 5396:5433 | NORMAL_TEXT | LIST id=kix.r7uwf2w9ada1 level=0]
identification of low-quality masks;

[P00106 | 5433:5467 | NORMAL_TEXT | LIST id=kix.r7uwf2w9ada1 level=0]
refinement using reliable frames;

[P00107 | 5467:5488 | NORMAL_TEXT | LIST id=kix.r7uwf2w9ada1 level=0]
PolypGen evaluation.

[P00108 | 5488:5693 | NORMAL_TEXT]
Critical difference: Their SQA score operates mainly as an external quality-control and re-segmentation mechanism. It does not appear to directly control the internal MedSAM2 memory update at every frame.

[P00109 | 5693:5724 | NORMAL_TEXT]
This gives you a possible gap:

[P00110 | 5724:5901 | NORMAL_TEXT]
Existing work detects unreliable masks and repairs them afterward, whereas your method can prevent unreliable predictions from contaminating the memory bank in the first place.

[P00111 | 5901:5974 | NORMAL_TEXT]
This paper must be treated as your nearest application-level competitor.

[P00112 | 5974:5976 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00113 | 5976:6042 | NORMAL_TEXT]
Phương pháp đánh giá với dữ liệu mới, chưa train, không liên quan

[P00114 | 6042:6165 | HEADING_3]
[5. SALI: Short-Term Alignment and Long-Term Interaction Network for Colonoscopy Video Polyp Segmentation — Hu et al., 2024](https://arxiv.org/pdf/2406.13532)

[P00115 | 6165:6194 | NORMAL_TEXT]
SALI specifically addresses:

[P00116 | 6194:6243 | NORMAL_TEXT | LIST id=kix.av5bh18vwbf5 level=0]
large changes between adjacent endoscopy frames;

[P00117 | 6243:6270 | NORMAL_TEXT | LIST id=kix.av5bh18vwbf5 level=0]
low-quality visual frames;

[P00118 | 6270:6300 | NORMAL_TEXT | LIST id=kix.av5bh18vwbf5 level=0]
long runs of degraded frames;

[P00119 | 6300:6336 | NORMAL_TEXT | LIST id=kix.av5bh18vwbf5 level=0]
reliable historical representation.

[P00120 | 6336:6579 | NORMAL_TEXT]
Its long-term interaction module stores historical polyp representations in a memory bank and uses them to reconstruct more reliable features when the current frame has weak visual cues. It reports improvements on SUN-SEG and has public code.

[P00121 | 6579:6588 | NORMAL_TEXT]
Overlap:

[P00122 | 6588:6614 | NORMAL_TEXT | LIST id=kix.fsequ4l27c8u level=0]
video polyp segmentation;

[P00123 | 6614:6642 | NORMAL_TEXT | LIST id=kix.fsequ4l27c8u level=0]
low-quality-frame handling;

[P00124 | 6642:6660 | NORMAL_TEXT | LIST id=kix.fsequ4l27c8u level=0]
long-term memory;

[P00125 | 6660:6697 | NORMAL_TEXT | LIST id=kix.fsequ4l27c8u level=0]
reliable historical representations.

[P00126 | 6697:6709 | NORMAL_TEXT]
Difference:

[P00127 | 6709:6766 | NORMAL_TEXT | LIST id=kix.7aj56rpmtr6t level=0]
It is a dedicated video-polyp architecture, not MedSAM2.

[P00128 | 6766:6875 | NORMAL_TEXT | LIST id=kix.7aj56rpmtr6t level=0]
It uses learned feature reconstruction rather than an explicit reliability-calibrated memory admission gate.

[P00129 | 6875:6943 | NORMAL_TEXT | LIST id=kix.7aj56rpmtr6t level=0]
It does not provide prompt-guided or interactive MedSAM2 behaviour.

[P00130 | 6943:7016 | NORMAL_TEXT]
Read for: how endoscopy-specific temporal problems should be formulated.

[P00131 | 7016:7018 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00132 | 7018:7115 | HEADING_3]
6. [Memory-Augmented SAM2 for Training-Free Surgical Video Segmentation — Yin et al., MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/2634_paper.pdf)

[P00133 | 7115:7288 | NORMAL_TEXT]
MA-SAM2 argues that SAM2’s greedy memory design becomes problematic in surgical videos because of occlusion, rapid motion and instrument–tissue interactions. It introduces:

[P00134 | 7288:7310 | NORMAL_TEXT | LIST id=kix.y7b5ilo550an level=0]
context-aware memory;

[P00135 | 7310:7338 | NORMAL_TEXT | LIST id=kix.y7b5ilo550an level=0]
occlusion-resilient memory;

[P00136 | 7338:7373 | NORMAL_TEXT | LIST id=kix.y7b5ilo550an level=0]
training-free memory augmentation;

[P00137 | 7373:7409 | NORMAL_TEXT | LIST id=kix.y7b5ilo550an level=0]
one-prompt long-video segmentation.

[P00138 | 7409:7477 | NORMAL_TEXT]
The method has code and was evaluated on EndoVis surgical datasets.

[P00139 | 7477:7486 | NORMAL_TEXT]
Overlap:

[P00140 | 7486:7517 | NORMAL_TEXT | LIST id=kix.xcxoy9cbspu8 level=0]
directly modifies SAM2 memory;

[P00141 | 7517:7543 | NORMAL_TEXT | LIST id=kix.xcxoy9cbspu8 level=0]
medical/endoscopic video;

[P00142 | 7543:7579 | NORMAL_TEXT | LIST id=kix.xcxoy9cbspu8 level=0]
reliability under difficult frames;

[P00143 | 7579:7609 | NORMAL_TEXT | LIST id=kix.xcxoy9cbspu8 level=0]
training-free implementation.

[P00144 | 7609:7621 | NORMAL_TEXT]
Difference:

[P00145 | 7621:7659 | NORMAL_TEXT | LIST id=kix.91gdzkdqmdw7 level=0]
focuses on instruments and occlusion;

[P00146 | 7659:7706 | NORMAL_TEXT | LIST id=kix.91gdzkdqmdw7 level=0]
does not focus on mask confidence calibration;

[P00147 | 7706:7785 | NORMAL_TEXT | LIST id=kix.91gdzkdqmdw7 level=0]
does not explicitly combine prompt agreement, image quality and mask dynamics;

[P00148 | 7785:7809 | NORMAL_TEXT | LIST id=kix.91gdzkdqmdw7 level=0]
not polyp segmentation.

[P00149 | 7809:7860 | NORMAL_TEXT]
Read for: practical SAM2 memory-bank modification.

[P00150 | 7860:7862 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00151 | 7862:7988 | HEADING_3]
[7. ARTEMIS: Agent-Guided Reliability-Aware Temporal Mask Evolution for Imperfectly Supervised Video Polyp Segmentation — 2026](https://arxiv.org/pdf/2606.20161)

[P00152 | 7988:8094 | NORMAL_TEXT]
This is the most important recent paper to examine because its title and framing are very close to yours.

[P00153 | 8094:8103 | NORMAL_TEXT]
ARTEMIS:

[P00154 | 8103:8138 | NORMAL_TEXT | LIST id=kix.w31t01vu8bv2 level=0]
performs video polyp segmentation;

[P00155 | 8138:8182 | NORMAL_TEXT | LIST id=kix.w31t01vu8bv2 level=0]
selects reliable masks as temporal anchors;

[P00156 | 8182:8260 | NORMAL_TEXT | LIST id=kix.w31t01vu8bv2 level=0]
avoids propagating all coarse masks because this may contaminate SAM2 memory;

[P00157 | 8260:8335 | NORMAL_TEXT | LIST id=kix.w31t01vu8bv2 level=0]
assigns anchor reliability using a vision-language debate-and-judge agent;

[P00158 | 8335:8391 | NORMAL_TEXT | LIST id=kix.w31t01vu8bv2 level=0]
uses reliable masks for bidirectional SAM2 propagation;

[P00159 | 8391:8445 | NORMAL_TEXT | LIST id=kix.w31t01vu8bv2 level=0]
trains with temporal reliability-aware pseudo labels.

[P00160 | 8445:8545 | NORMAL_TEXT]
The authors explicitly state that propagating every coarse mask may contaminate SAM2’s memory bank.

[P00161 | 8545:8594 | NORMAL_TEXT]
Overlap: Extremely high at the conceptual level.

[P00162 | 8594:8634 | NORMAL_TEXT]
Potential differences you can preserve:

[P00163 | 8634:8714 | NORMAL_TEXT | LIST id=kix.izv2rl5w75vl level=0]
ARTEMIS uses a relatively expensive vision-language agent for anchor decisions.

[P00164 | 8714:8779 | NORMAL_TEXT | LIST id=kix.izv2rl5w75vl level=0]
It focuses on imperfect supervision and pseudo-label refinement.

[P00165 | 8779:8830 | NORMAL_TEXT | LIST id=kix.izv2rl5w75vl level=0]
Your system can be lightweight and inference-time.

[P00166 | 8830:8920 | NORMAL_TEXT | LIST id=kix.izv2rl5w75vl level=0]
Your decision can be grounded in calibrated numerical signals rather than agent judgment.

[P00167 | 8920:9026 | NORMAL_TEXT | LIST id=kix.izv2rl5w75vl level=0]
Your target action can be explicit three-way memory control: accept, reject/freeze, or replace/re-prompt.

[P00168 | 9026:9115 | NORMAL_TEXT | LIST id=kix.izv2rl5w75vl level=0]
Your evaluation can focus on memory contamination, drift duration and correction burden.

[P00169 | 9115:9185 | NORMAL_TEXT]
Do not finalize your contribution before studying this paper closely.

[P00170 | 9185:9187 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00171 | 9187:9309 | HEADING_3]
[8. Strike the Balance: On-the-Fly Uncertainty-Based User Interactions for Long-Term Video Object Segmentation — ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/papers/Vujasinovic_Strike_the_Balance_On-the-Fly_Uncertainty_based_User_Interactions_for_Long-Term_ACCV_2024_paper.pdf?utm_source=chatgpt.com)

[P00172 | 9309:9550 | NORMAL_TEXT]
This work introduces uncertainty-triggered user interaction for long-term video object segmentation. It discusses a memory model in which updates depend on whether confidence is high enough for the predicted mask to be regarded as reliable.

[P00173 | 9550:9559 | NORMAL_TEXT]
Overlap:

[P00174 | 9559:9582 | NORMAL_TEXT | LIST id=kix.vv21o3w1jxep level=0]
reliability threshold;

[P00175 | 9582:9608 | NORMAL_TEXT | LIST id=kix.vv21o3w1jxep level=0]
uncertainty-aware update;

[P00176 | 9608:9631 | NORMAL_TEXT | LIST id=kix.vv21o3w1jxep level=0]
long-term propagation;

[P00177 | 9631:9666 | NORMAL_TEXT | LIST id=kix.vv21o3w1jxep level=0]
triggering corrective interaction.

[P00178 | 9666:9678 | NORMAL_TEXT]
Difference:

[P00179 | 9678:9743 | NORMAL_TEXT | LIST id=kix.lpo5linyw460 level=0]
general video object segmentation rather than medical endoscopy;

[P00180 | 9743:9756 | NORMAL_TEXT | LIST id=kix.lpo5linyw460 level=0]
not MedSAM2;

[P00181 | 9756:9805 | NORMAL_TEXT | LIST id=kix.lpo5linyw460 level=0]
Human correction is a major part of the setting.

[P00182 | 9805:9866 | NORMAL_TEXT]
This is useful for designing your correction-trigger policy.

[P00183 | 9866:9868 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00184 | 9868:9943 | HEADING_3]
[9. SAM2Long: Enhancing SAM2 for Long Video Object Segmentation — 2024/2025](https://arxiv.org/pdf/2410.16268)

[P00185 | 9943:9976 | NORMAL_TEXT]
Occlusions + Object reappearance

[P00186 | 9976:10132 | NORMAL_TEXT]
SAM2Long models frame-level segmentation uncertainty and keeps multiple propagation hypotheses, selecting a video-level path using constrained tree search.

[P00187 | 10132:10141 | NORMAL_TEXT]
Overlap:

[P00188 | 10141:10183 | NORMAL_TEXT | LIST id=kix.til0adyu1wuo level=0]
acknowledges SAM2 prediction uncertainty;

[P00189 | 10183:10249 | NORMAL_TEXT | LIST id=kix.til0adyu1wuo level=0]
prevents a single early error from dominating future propagation;

[P00190 | 10249:10314 | NORMAL_TEXT | LIST id=kix.til0adyu1wuo level=0]
improves long-video robustness without conventional fine-tuning.

[P00191 | 10314:10326 | NORMAL_TEXT]
Difference:

[P00192 | 10326:10392 | NORMAL_TEXT | LIST id=kix.kugdy4lcjmct level=0]
maintains multiple pathways rather than filtering memory updates;

[P00193 | 10392:10405 | NORMAL_TEXT | LIST id=kix.kugdy4lcjmct level=0]
general VOS;

[P00194 | 10405:10468 | NORMAL_TEXT | LIST id=kix.kugdy4lcjmct level=0]
higher computational cost than a lightweight reliability gate.

[P00195 | 10468:10529 | NORMAL_TEXT]
Read for: alternatives to binary accept/reject memory logic.

[P00196 | 10529:10538 | NORMAL_TEXT]
Problem:

[P00197 | 10538:10548 | NORMAL_TEXT | LIST id=kix.t7qp8c9l7w3s level=0]
Scenario:

[P00198 | 10548:10579 | NORMAL_TEXT | LIST id=kix.t7qp8c9l7w3s level=1]
Long term video tracking video

[P00199 | 10579:10621 | NORMAL_TEXT | LIST id=kix.t7qp8c9l7w3s level=1]
frequent occlusions + reappearing objects

[P00200 | 10621:10686 | NORMAL_TEXT | LIST id=kix.t7qp8c9l7w3s level=1]
when SAM2 does good, also MedSAM2: clear visual cues are present

[P00201 | 10686:10864 | NORMAL_TEXT | LIST id=kix.2hzsvu776i0i level=0]
greedy selection memory design (only 1 mask with the highest predicted IoU in each frame) => error accumulation (errored/missed masks influence segmentation of Subsequent frame)

[P00202 | 10864:10877 | NORMAL_TEXT]
Inspiration:

[P00203 | 10877:10983 | NORMAL_TEXT | LIST id=kix.gzei971k7aqm level=0]
Observations: SAM2 mask decoder generate multiple diverse masks + IoU predicted scores + occlusions score

[P00204 | 10983:11021 | NORMAL_TEXT | LIST id=kix.obnv84sn9siq level=0]
Multiple Hypothesis Tracking (backup)

[P00205 | 11021:11087 | NORMAL_TEXT | LIST id=kix.obnv84sn9siq level=1]
fixed number of memory pathways => explore multiple segmentations

[P00206 | 11087:11092 | NORMAL_TEXT | LIST id=kix.obnv84sn9siq level=1]
XMEM

[P00207 | 11092:11111 | NORMAL_TEXT | LIST id=kix.obnv84sn9siq level=1]
tree constrained: 

[P00208 | 11111:11180 | NORMAL_TEXT | LIST id=kix.obnv84sn9siq level=2]
accumulated logarithm of the predicted IoU scores across the pathway

[P00209 | 11180:11266 | NORMAL_TEXT | LIST id=kix.obnv84sn9siq level=2]
select pathways with distinct predicted masks, occlusion scores indicate uncertainty.

[P00210 | 11266:11274 | NORMAL_TEXT]
Method:

[P00211 | 11274:11302 | NORMAL_TEXT | LIST id=kix.xlm2vslcckxy level=0]
cumulative score algorithm:

[P00212 | 11302:11304 | NORMAL_TEXT]
[INLINE_OBJECT kix.x4d91uj1glr8]

[P00213 | 11304:11322 | NORMAL_TEXT | LIST id=kix.f5we5x6dmpcb level=0]
Memory attention:

[P00214 | 11322:11325 | NORMAL_TEXT]
[INLINE_OBJECT kix.1ps5n4xxwkhg][INLINE_OBJECT kix.p1zdj2hwtat6]

[P00215 | 11325:11339 | NORMAL_TEXT]
Contribution:

[P00216 | 11339:11376 | NORMAL_TEXT | LIST id=kix.g6mhgckvgwcg level=0]
training-free segmentation strategy:

[P00217 | 11376:11415 | NORMAL_TEXT | LIST id=kix.g6mhgckvgwcg level=1]
consider Uncertainty within each frame

[P00218 | 11415:11507 | NORMAL_TEXT | LIST id=kix.g6mhgckvgwcg level=1]
video-level optimal results from multiple segmentation pathways in constrained tree search.

[P00219 | 11507:11531 | NORMAL_TEXT | LIST id=kix.g6mhgckvgwcg level=1]
heuristic search design

[P00220 | 11531:11613 | NORMAL_TEXT | LIST id=kix.g6mhgckvgwcg level=0]
ensure memory bank provide effective object cues for current frame’s segmentation

[P00221 | 11613:11777 | NORMAL_TEXT | LIST id=kix.g6mhgckvgwcg level=0]
modulate the memory attention calculation by weighting memory entries according to their occlusion scores, emphasizing more reliable entries during crossattention.

[P00222 | 11777:11823 | NORMAL_TEXT | LIST id=kix.g6mhgckvgwcg level=0]
robust toward occlusions + object reappearing

[P00223 | 11823:11836 | NORMAL_TEXT]
Limitations:

[P00224 | 11836:11947 | NORMAL_TEXT | LIST id=kix.26yydkwhu5ob level=0]
between the selections and the weighted fusing of the memory, which drives the performance of the whole model?

[P00225 | 11947:11949 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00226 | 11949:12065 | HEADING_3]
[10. Inference-Time Temporal Probability Smoothing for Stable Video Segmentation with SAM2 under Weak Prompts — 2026](https://arxiv.org/pdf/2604.17115)

[P00227 | 12065:12087 | NORMAL_TEXT]
This method combines:

[P00228 | 12087:12114 | NORMAL_TEXT | LIST id=kix.sxnxxepkuh23 level=0]
entropy-based uncertainty;

[P00229 | 12114:12136 | NORMAL_TEXT | LIST id=kix.sxnxxepkuh23 level=0]
optical-flow warping;

[P00230 | 12136:12171 | NORMAL_TEXT | LIST id=kix.sxnxxepkuh23 level=0]
forward–backward flow consistency;

[P00231 | 12171:12216 | NORMAL_TEXT | LIST id=kix.sxnxxepkuh23 level=0]
adaptive blending with previous predictions.

[P00232 | 12216:12363 | NORMAL_TEXT]
It is lightweight and inference-time, but it operates mainly on the output probability maps rather than fundamentally controlling the memory bank.

[P00233 | 12363:12554 | NORMAL_TEXT]
This paper is useful for distinguishing output smoothing from reliability-aware memory control. That distinction directly matches the weakness you previously identified in your current gate.

[P00234 | 12554:12563 | NORMAL_TEXT]
Context:

[P00235 | 12563:12625 | NORMAL_TEXT | LIST id=kix.746t37hwy73b level=0]
weak user supervision - sparse prompts - temporal instability

[P00236 | 12625:12670 | NORMAL_TEXT | LIST id=kix.746t37hwy73b level=0]
produce temporarily inconsistent predictions

[P00237 | 12670:12678 | NORMAL_TEXT]
Method:

[P00238 | 12678:12680 | NORMAL_TEXT]
[INLINE_OBJECT kix.403lbk6qwvi5]

[P00239 | 12680:12712 | NORMAL_TEXT | LIST id=kix.xkrsiqx51agp level=0]
Pixel-wise entropy uncertainty:

[P00240 | 12712:12714 | NORMAL_TEXT]
[INLINE_OBJECT kix.is1wdm8tmx7]

[P00241 | 12714:12716 | NORMAL_TEXT]
[INLINE_OBJECT kix.yzlo38wqaolz]

[P00242 | 12716:12730 | NORMAL_TEXT]
Contribution:

[P00243 | 12730:12783 | NORMAL_TEXT | LIST id=kix.wso28s4i6tc level=0]
inference-time temporal probability smoothing method

[P00244 | 12783:12920 | NORMAL_TEXT | LIST id=kix.wso28s4i6tc level=0]
per-frame segmentation probability masks + leverage optical-flow-based motion warping + pixel-wise uncertainty from segmentation entropy

[P00245 | 12920:13031 | NORMAL_TEXT | LIST id=kix.wso28s4i6tc level=0]
adaptively BLEND current-frame predictions with motion-aligned historical estimates => coherent between frames

[P00246 | 13031:13044 | NORMAL_TEXT]
Limitations:

[P00247 | 13044:13124 | NORMAL_TEXT | LIST id=kix.38fqd7p51nsx level=0]
the sparse prompt rate (prompt stride) is not explicitly and carefully analysed

[P00248 | 13124:13256 | NORMAL_TEXT | LIST id=kix.ckmpctjcfem5 level=0]
It creates a new evaluation metrics => hard to compare with other methods using the popular DICE, IoU, Precision, Recall, F2 scores

[P00249 | 13256:13324 | HEADING_3]
[11. SurgSAM-2: Surgical SAM 2 — Liu et al., 2024](https://arxiv.org/pdf/2408.07931) (not need to care)

[P00250 | 13324:13428 | NORMAL_TEXT]
Why it is essential: It directly modifies SAM2’s memory bank for efficient surgical-video segmentation.

[P00251 | 13428:13481 | NORMAL_TEXT]
The paper introduces Efficient Frame Pruning, which:

[P00252 | 13481:13531 | NORMAL_TEXT | LIST id=kix.6eit9xck5on1 level=0]
compares memory features using cosine similarity;

[P00253 | 13531:13573 | NORMAL_TEXT | LIST id=kix.6eit9xck5on1 level=0]
removes highly similar, redundant frames;

[P00254 | 13573:13617 | NORMAL_TEXT | LIST id=kix.6eit9xck5on1 level=0]
preserves diverse and informative memories;

[P00255 | 13617:13673 | NORMAL_TEXT | LIST id=kix.6eit9xck5on1 level=0]
reduces memory use and increases inference speed.[https://arxiv.org/abs/2408.07931?utm_source=chatgpt.com](https://arxiv.org/abs/2408.07931?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2408.07931?utm_source=chatgpt.com)

[P00256 | 13673:13738 | NORMAL_TEXT]
Overlap with your idea: Moderate for adaptive memory management.

[P00257 | 13738:13782 | NORMAL_TEXT]
Difference from your intended contribution:

[P00258 | 13782:13829 | NORMAL_TEXT | LIST id=kix.d7yhkkm1qzzl level=0]
It focuses mainly on computational efficiency.

[P00259 | 13829:13900 | NORMAL_TEXT | LIST id=kix.d7yhkkm1qzzl level=0]
Memory selection is based on feature similarity, not mask reliability.

[P00260 | 13900:13962 | NORMAL_TEXT | LIST id=kix.d7yhkkm1qzzl level=0]
A corrupted but visually different frame may still be stored.

[P00261 | 13962:14037 | NORMAL_TEXT | LIST id=kix.d7yhkkm1qzzl level=0]
It does not use image quality, prompt agreement or calibrated uncertainty.

[P00262 | 14037:14114 | NORMAL_TEXT]
Read for: similarity-based memory pruning and efficient SAM2 implementation.

[P00263 | 14114:14123 | NORMAL_TEXT]
Problem:

[P00264 | 14123:14165 | NORMAL_TEXT | LIST id=kix.1cll2tnribd6 level=0]
high computational demands (not relevant)

[P00265 | 14165:14204 | NORMAL_TEXT | LIST id=kix.1cll2tnribd6 level=0]
complex + long-range temporal dynamics

[P00266 | 14204:14220 | NORMAL_TEXT | LIST id=kix.1cll2tnribd6 level=0]
needs real-time

[P00267 | 14220:14241 | NORMAL_TEXT | LIST id=kix.1cll2tnribd6 level=0]
SAM2’s memory bank: 

[P00268 | 14241:14326 | NORMAL_TEXT | LIST id=kix.1cll2tnribd6 level=1]
store frames Sequentially => retains redundant information => inflate (not relevant)

[P00269 | 14326:14366 | NORMAL_TEXT | LIST id=kix.1cll2tnribd6 level=1]
first-come-first-serve memory mechanism

[P00270 | 14366:14380 | NORMAL_TEXT]
Inspirations:

[P00271 | 14380:14391 | NORMAL_TEXT | LIST id=kix.aplr9hd7lb6j level=0]
XMEM, RMEM

[P00272 | 14391:14406 | NORMAL_TEXT]
Contributions:

[P00273 | 14406:14441 | NORMAL_TEXT | LIST id=kix.iq0jb5dfrjr1 level=0]
dynamically manage the memory bank

[P00274 | 14441:14488 | NORMAL_TEXT | LIST id=kix.iq0jb5dfrjr1 level=1]
selectively choose the most informative frames

[P00275 | 14488:14497 | NORMAL_TEXT | LIST id=kix.iq0jb5dfrjr1 level=1]
PRUNING:

[P00276 | 14497:14552 | NORMAL_TEXT | LIST id=kix.iq0jb5dfrjr1 level=2]
cosine similarity - retain only the INFORMATIVE frames

[P00277 | 14552:14586 | NORMAL_TEXT | LIST id=kix.iq0jb5dfrjr1 level=0]
resource-constrained environments

[P00278 | 14586:14588 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00279 | 14588:14620 | HEADING_3]
[12. TSMS-SAM2 — Xu et al., 2025](https://arxiv.org/pdf/2508.05829)

[P00280 | 14620:14734 | NORMAL_TEXT]
Why it is essential: It improves SAM2 memory organization for rapid motion and long-term surgical-video tracking.

[P00281 | 14734:14756 | NORMAL_TEXT]
The paper introduces:

[P00282 | 14756:14803 | NORMAL_TEXT | LIST id=kix.6np89kyvmn42 level=0]
multi-scale temporal sampling during training;

[P00283 | 14803:14845 | NORMAL_TEXT | LIST id=kix.6np89kyvmn42 level=0]
separate short-term and long-term memory;

[P00284 | 14845:14896 | NORMAL_TEXT | LIST id=kix.6np89kyvmn42 level=0]
similarity-based pruning within each memory group;

[P00285 | 14896:14947 | NORMAL_TEXT | LIST id=kix.6np89kyvmn42 level=0]
preservation of diverse historical features.[https://arxiv.org/abs/2508.05829?utm_source=chatgpt.com](https://arxiv.org/abs/2508.05829?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2508.05829?utm_source=chatgpt.com)

[P00286 | 14947:15020 | NORMAL_TEXT]
Overlap with your idea: Moderate to high for structured adaptive memory.

[P00287 | 15020:15064 | NORMAL_TEXT]
Difference from your intended contribution:

[P00288 | 15064:15103 | NORMAL_TEXT | LIST id=kix.u4uy3wp5gem9 level=0]
It organizes memory by temporal range.

[P00289 | 15103:15169 | NORMAL_TEXT | LIST id=kix.u4uy3wp5gem9 level=0]
It removes redundant features rather than unreliable predictions.

[P00290 | 15169:15223 | NORMAL_TEXT | LIST id=kix.u4uy3wp5gem9 level=0]
It requires model training and temporal augmentation.

[P00291 | 15223:15310 | NORMAL_TEXT | LIST id=kix.u4uy3wp5gem9 level=0]
It does not decide whether the current mask should be completely rejected from memory.

[P00292 | 15310:15380 | NORMAL_TEXT | LIST id=kix.u4uy3wp5gem9 level=0]
It does not calibrate confidence against actual segmentation quality.

[P00293 | 15380:15454 | NORMAL_TEXT]
Read for: short-/long-term memory design and motion-robust SAM2 training.

[P00294 | 15454:15463 | NORMAL_TEXT]
Problem:

[P00295 | 15463:15510 | NORMAL_TEXT | LIST id=kix.hvlqnrmdm6d9 level=0]
complex motion dynamics + redundancy of memory

[P00296 | 15510:15540 | NORMAL_TEXT | LIST id=kix.hvlqnrmdm6d9 level=0]
disrupted temporal continuity

[P00297 | 15540:15561 | NORMAL_TEXT | LIST id=kix.hvlqnrmdm6d9 level=0]
SAM2’s memory bank: 

[P00298 | 15561:15623 | NORMAL_TEXT | LIST id=kix.hvlqnrmdm6d9 level=1]
introduce redundancy due to inherent continuity of video data

[P00299 | 15623:15631 | NORMAL_TEXT]
Method:

[P00300 | 15631:15670 | NORMAL_TEXT | LIST id=kix.qqoy06149wbc level=0]
memory splitting and pruning mechanism

[P00301 | 15670:15736 | NORMAL_TEXT | LIST id=kix.qqoy06149wbc level=1]
remove redundant features from previous frames in the memory bank

[P00302 | 15736:15751 | NORMAL_TEXT]
Contributions:

[P00303 | 15751:15864 | NORMAL_TEXT | LIST id=kix.jzfbub3njrmy level=0]
multi-temporal-scale (multi-stride) video sampling augmentation => improve robustness against motion variability

[P00304 | 15864:15961 | NORMAL_TEXT | LIST id=kix.jzfbub3njrmy level=0]
memory splitting and pruning => organise and filter past frame features - efficient segmentation

[P00305 | 15961:15974 | NORMAL_TEXT]
Limitations:

[P00306 | 15974:16014 | NORMAL_TEXT | LIST id=kix.xtr89bqcqd73 level=0]
similarity does NOT measure reliability

[P00307 | 16014:16041 | NORMAL_TEXT | LIST id=kix.xtr89bqcqd73 level=0]
reference-frame dependence

[P00308 | 16041:16075 | NORMAL_TEXT | LIST id=kix.xtr89bqcqd73 level=1]
oldest frame for long-term memory

[P00309 | 16075:16110 | NORMAL_TEXT | LIST id=kix.xtr89bqcqd73 level=1]
latest frame for short-term memory

[P00310 | 16110:16236 | NORMAL_TEXT | LIST id=kix.xtr89bqcqd73 level=0]
No ablations - not clear about the influence of the multi-scale augmentation and pruning mechanism to the model’s performance

[P00311 | 16236:16238 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00312 | 16238:16268 | HEADING_3]
[13. FreeVPS — Hu et al., 2025](https://arxiv.org/pdf/2508.19705)

[P00313 | 16268:16392 | NORMAL_TEXT]
Why it is essential: This is the closest paper to your problem of preventing SAM2 memory drift in video polyp segmentation.

[P00314 | 16392:16432 | NORMAL_TEXT]
FreeVPS uses two training-free modules:

[P00315 | 16432:16519 | NORMAL_TEXT | LIST id=kix.fwb3h5byxzil level=0]
Intra-association filtering: removes temporally inconsistent image-segmentation masks;

[P00316 | 16519:16633 | NORMAL_TEXT | LIST id=kix.fwb3h5byxzil level=0]
Inter-association refinement: compares image-model masks with SAM2 tracks and adaptively updates the memory bank.

[P00317 | 16633:16812 | NORMAL_TEXT]
Its purpose is to prevent the snowball effect, where incorrect masks enter memory and damage later predictions. It is evaluated on video-polyp datasets, including PolypGen.[https://arxiv.org/abs/2508.19705?utm_source=chatgpt.com](https://arxiv.org/abs/2508.19705?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2508.19705?utm_source=chatgpt.com)

[P00318 | 16812:16896 | NORMAL_TEXT]
Overlap with your idea: Very high for adaptive memory updates and drift prevention.

[P00319 | 16896:16940 | NORMAL_TEXT]
Difference from your intended contribution:

[P00320 | 16940:16996 | NORMAL_TEXT | LIST id=kix.oukw5pwiy6vl level=0]
It requires an external image-polyp segmentation model.

[P00321 | 16996:17070 | NORMAL_TEXT | LIST id=kix.oukw5pwiy6vl level=0]
Reliability is estimated through temporal agreement and mask association.

[P00322 | 17070:17136 | NORMAL_TEXT | LIST id=kix.oukw5pwiy6vl level=0]
It does not produce a calibrated probability of mask correctness.

[P00323 | 17136:17206 | NORMAL_TEXT | LIST id=kix.oukw5pwiy6vl level=0]
It does not explicitly analyse blur, darkness or specular highlights.

[P00324 | 17206:17302 | NORMAL_TEXT | LIST id=kix.oukw5pwiy6vl level=0]
It mainly uses update and trajectory-removal rules rather than update/freeze/re-prompt actions.

[P00325 | 17302:17386 | NORMAL_TEXT]
Read for: adaptive SAM2 memory updates, temporal consensus and PolypGen comparison.

[P00326 | 17386:17395 | NORMAL_TEXT]
Problem:

[P00327 | 17395:17473 | NORMAL_TEXT | LIST id=kix.8pvag57wzv9f level=0]
struggle to balance between spatiotemporal modeling and domain generalisation

[P00328 | 17473:17492 | NORMAL_TEXT | LIST id=kix.8pvag57wzv9f level=0]
error accumulation

[P00329 | 17492:17500 | NORMAL_TEXT]
Method:

[P00330 | 17500:17511 | NORMAL_TEXT | LIST id=kix.pdbpxxpr1qdi level=0]
Detecting:

[P00331 | 17511:17547 | NORMAL_TEXT | LIST id=kix.pdbpxxpr1qdi level=1]
foundations => reliable predictions

[P00332 | 17547:17557 | NORMAL_TEXT | LIST id=kix.pdbpxxpr1qdi level=0]
Tracking:

[P00333 | 17557:17602 | NORMAL_TEXT | LIST id=kix.pdbpxxpr1qdi level=1]
strengthen spatiotemporal coherence of masks

[P00334 | 17602:17616 | NORMAL_TEXT]
Contribution:

[P00335 | 17616:17633 | NORMAL_TEXT | LIST id=kix.387wk49gface level=0]
track by detect:

[P00336 | 17633:17678 | NORMAL_TEXT | LIST id=kix.387wk49gface level=1]
spatial contexts in image polyp segmentation

[P00337 | 17678:17717 | NORMAL_TEXT | LIST id=kix.387wk49gface level=1]
temporal modeling capabilities of SAM2

[P00338 | 17717:17728 | NORMAL_TEXT | LIST id=kix.387wk49gface level=0]
2 modules:

[P00339 | 17728:17855 | NORMAL_TEXT | LIST id=kix.387wk49gface level=1]
intra-association filtering - IPS (image polyp segmentation) - discover temporally consistent predictions over multiple frames

[P00340 | 17855:17985 | NORMAL_TEXT | LIST id=kix.387wk49gface level=1]
inter-association refinement - update memory bank to prevent error propagation => temporal coherence - refine propagation results

[P00341 | 17985:17998 | NORMAL_TEXT]
Limitations:

[P00342 | 17998:18080 | NORMAL_TEXT | LIST id=kix.2otayw7a8t6q level=0]
strongly depend on reliable image polyp segmenter. Especially the reference image

[P00343 | 18080:18082 | NORMAL_TEXT]
[INLINE_OBJECT kix.gihnuti0ge2x]

[P00344 | 18082:18107 | NORMAL_TEXT | LIST id=kix.2otayw7a8t6q level=0]
agreement != correctness

[P00345 | 18107:18109 | NORMAL_TEXT]
[INLINE_OBJECT kix.pu04h3stzun5]

[P00346 | 18109:18142 | NORMAL_TEXT | LIST id=kix.xalncsc5g6wf level=0]
offline, window-based refinement

[P00347 | 18142:18182 | NORMAL_TEXT | LIST id=kix.xalncsc5g6wf level=0]
Fixed temporal window: insufficient for

[P00348 | 18182:18206 | NORMAL_TEXT | LIST id=kix.xalncsc5g6wf level=1]
gradual long-term drift

[P00349 | 18206:18224 | NORMAL_TEXT | LIST id=kix.xalncsc5g6wf level=1]
long blur periods

[P00350 | 18224:18245 | NORMAL_TEXT | LIST id=kix.xalncsc5g6wf level=1]
prolonged occlusions

[P00351 | 18245:18269 | NORMAL_TEXT | LIST id=kix.xalncsc5g6wf level=1]
irregular camera motion

[P00352 | 18269:18299 | NORMAL_TEXT | LIST id=kix.xalncsc5g6wf level=1]
slowly (dis)appearing objects

[P00353 | 18299:18324 | NORMAL_TEXT | LIST id=kix.xalncsc5g6wf level=0]
Added computational cost

[P00354 | 18324:18424 | HEADING_3]
[14. ReMeDI-SAM3: Refined Memory for Disambiguation of Identities with SAM3 in Surgical Segmentation](https://arxiv.org/pdf/2512.16880)

[P00355 | 18424:18482 | NORMAL_TEXT]
This is currently the closest SAM3 paper to your problem.

[P00356 | 18482:18507 | NORMAL_TEXT]
It argues that SAM3 has:

[P00357 | 18507:18562 | NORMAL_TEXT | LIST id=kix.jnmkp0jtqbx7 level=0]
reliability-agnostic or indiscriminate memory updates;

[P00358 | 18562:18613 | NORMAL_TEXT | LIST id=kix.jnmkp0jtqbx7 level=0]
memory contamination from low-quality predictions;

[P00359 | 18613:18649 | NORMAL_TEXT | LIST id=kix.jnmkp0jtqbx7 level=0]
error accumulation after occlusion;

[P00360 | 18649:18693 | NORMAL_TEXT | LIST id=kix.jnmkp0jtqbx7 level=0]
identity drift during long surgical videos.

[P00361 | 18693:18716 | NORMAL_TEXT]
Its solution includes:

[P00362 | 18716:18750 | NORMAL_TEXT | LIST id=kix.a855pwp3t4js level=0]
relevance-aware memory filtering;

[P00363 | 18750:18785 | NORMAL_TEXT | LIST id=kix.a855pwp3t4js level=0]
a separate occlusion-aware memory;

[P00364 | 18785:18820 | NORMAL_TEXT | LIST id=kix.a855pwp3t4js level=0]
confidence-based memory admission;

[P00365 | 18820:18847 | NORMAL_TEXT | LIST id=kix.a855pwp3t4js level=0]
memory-capacity expansion;

[P00366 | 18847:18901 | NORMAL_TEXT | LIST id=kix.a855pwp3t4js level=0]
feature-based re-identification with temporal voting.

[P00367 | 18901:18976 | NORMAL_TEXT]
It is training-free and evaluated on EndoVis17, EndoVis18 and CholecSeg8k.

[P00368 | 18976:19010 | NORMAL_TEXT]
Overlap with your work: very high

[P00369 | 19010:19091 | NORMAL_TEXT]
Both methods aim to prevent unreliable frames from damaging future segmentation.

[P00370 | 19091:19109 | NORMAL_TEXT]
Main differences:

[P00371 | 19112:19124 | NORMAL_TEXT | TABLE row=0 col=0]
ReMeDI-SAM3

[P00372 | 19125:19149 | NORMAL_TEXT | TABLE row=0 col=1]
Your intended direction

[P00373 | 19151:19184 | NORMAL_TEXT | TABLE row=1 col=0]
Surgical instrument segmentation

[P00374 | 19185:19229 | NORMAL_TEXT | TABLE row=1 col=1]
Polyp and later adenoid/airway segmentation

[P00375 | 19231:19275 | NORMAL_TEXT | TABLE row=2 col=0]
Focuses on occlusion, re-entry and identity

[P00376 | 19276:19343 | NORMAL_TEXT | TABLE row=2 col=1]
Focuses on blur, specular highlights, prompt errors and mask drift

[P00377 | 19345:19386 | NORMAL_TEXT | TABLE row=3 col=0]
Confidence and occlusion-aware filtering

[P00378 | 19387:19423 | NORMAL_TEXT | TABLE row=3 col=1]
Multisignal reliability calibration

[P00379 | 19425:19470 | NORMAL_TEXT | TABLE row=4 col=0]
Re-identification after object disappearance

[P00380 | 19471:19531 | NORMAL_TEXT | TABLE row=4 col=1]
Freeze/reset/re-prompt after unreliable tissue segmentation

[P00381 | 19533:19568 | NORMAL_TEXT | TABLE row=5 col=0]
Multi-instrument identity tracking

[P00382 | 19569:19604 | NORMAL_TEXT | TABLE row=5 col=1]
Anatomical or lesion mask accuracy

[P00383 | 19605:19658 | NORMAL_TEXT]
Therefore, your contribution must not broadly claim:

[P00384 | 19658:19715 | NORMAL_TEXT]
“The first reliability-aware memory mechanism for SAM3.”

[P00385 | 19715:19756 | NORMAL_TEXT]
ReMeDI-SAM3 already occupies that space.

[P00386 | 19756:19790 | NORMAL_TEXT]
A more defensible claim would be:

[P00387 | 19790:19971 | NORMAL_TEXT]
“An endoscopy-specific, reliability-calibrated memory controller for SAM3/MedSAM3 targeting deformable anatomical and lesion segmentation rather than instrument identity tracking.”

[P00388 | 19971:20048 | HEADING_3]
[15. SAM3-DMS: Decoupled Memory Selection for Multi-target Video Segmentation](https://arxiv.org/pdf/2601.09699)

[P00389 | 20048:20374 | NORMAL_TEXT]
SAM3-DMS identifies a weakness in SAM3’s original memory-selection policy. SAM3 may average confidence across several objects and make a synchronized memory-update decision. Consequently, one poorly segmented object may still have an incorrect or blank mask written into memory because the other objects have high confidence.

[P00390 | 20374:20492 | NORMAL_TEXT]
SAM3-DMS changes this to object-specific memory decisions using each target’s segmentation confidence and visibility.

[P00391 | 20492:20579 | NORMAL_TEXT]
This is directly relevant to your future adenoid problem because you have two targets:

[P00392 | 20579:20588 | NORMAL_TEXT | LIST id=kix.v4xzsg1im3sz level=0]
adenoid;

[P00393 | 20588:20611 | NORMAL_TEXT | LIST id=kix.v4xzsg1im3sz level=0]
nasopharyngeal airway.

[P00394 | 20611:20831 | NORMAL_TEXT]
A shared group-level decision may be inappropriate. The adenoid may be clear while the airway is occluded or poorly exposed. Each object should therefore have an independent reliability score and memory-update decision.

[P00395 | 20831:20891 | NORMAL_TEXT]
Overlap: individual reliability-conditioned memory updates.

[P00396 | 20891:21102 | NORMAL_TEXT]
Difference: SAM3-DMS is general multi-object VOS and mainly addresses synchronization across targets. It does not calibrate reliability using endoscopy image quality, anatomical consistency or prompt agreement.

[P00397 | 21102:21104 | NORMAL_TEXT]
[INLINE_OBJECT kix.f6hy9aubkc1s]

[P00398 | 21104:21134 | HEADING_2]
What SAM3 itself already does

[P00399 | 21134:21192 | NORMAL_TEXT]
SAM3 is not simply SAM2 with text prompting. It combines:

[P00400 | 21192:21233 | NORMAL_TEXT | LIST id=kix.m98n51cffp2o level=0]
an image-level open-vocabulary detector;

[P00401 | 21233:21263 | NORMAL_TEXT | LIST id=kix.m98n51cffp2o level=0]
a memory-based video tracker;

[P00402 | 21263:21303 | NORMAL_TEXT | LIST id=kix.m98n51cffp2o level=0]
text or image-exemplar concept prompts;

[P00403 | 21303:21331 | NORMAL_TEXT | LIST id=kix.m98n51cffp2o level=0]
object presence prediction;

[P00404 | 21331:21367 | NORMAL_TEXT | LIST id=kix.m98n51cffp2o level=0]
persistent identities across video.

[P00405 | 21367:21479 | NORMAL_TEXT]
Meta describes it as a model that detects, segments and tracks all instances matching a concept prompt.[https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)[Meta AI](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)

[P00406 | 21479:21723 | NORMAL_TEXT]
Importantly, the SAM3 literature indicates that it already includes a basic confidence-conditioned memory-selection mechanism. SAM3-DMS explains that SAM3 thresholds prediction confidence and memorizes only features regarded as reliable.[https://arxiv.org/html/2601.09699v1](https://arxiv.org/html/2601.09699v1)[arXiv](https://arxiv.org/html/2601.09699v1)

[P00407 | 21723:21817 | NORMAL_TEXT]
Therefore, for SAM3, you cannot simply propose: “Only update memory when confidence is high.”

[P00408 | 21817:21862 | NORMAL_TEXT]
That is already part of the baseline design.

[P00409 | 21862:21919 | NORMAL_TEXT]
You would need to improve at least one of these aspects:

[P00410 | 21919:21960 | NORMAL_TEXT | LIST id=kix.gpjvpcsfujqz level=0]
the quality of the reliability estimate;

[P00411 | 21960:21977 | NORMAL_TEXT | LIST id=kix.gpjvpcsfujqz level=0]
its calibration;

[P00412 | 21977:22004 | NORMAL_TEXT | LIST id=kix.gpjvpcsfujqz level=0]
object-specific decisions;

[P00413 | 22004:22040 | NORMAL_TEXT | LIST id=kix.gpjvpcsfujqz level=0]
endoscopy-specific failure signals;

[P00414 | 22040:22058 | NORMAL_TEXT | LIST id=kix.gpjvpcsfujqz level=0]
recovery actions;

[P00415 | 22058:22086 | NORMAL_TEXT | LIST id=kix.gpjvpcsfujqz level=0]
clinical correction burden.

[P00416 | 22086:22106 | HEADING_2]
What about MedSAM3?

[P00417 | 22106:22167 | HEADING_3]
[MedSAM3: Delving into Segment Anything with Medical Concepts](https://arxiv.org/pdf/2511.19046)

[P00418 | 22167:22413 | NORMAL_TEXT]
The original MedSAM3 paper adapts SAM3 for text-promptable medical image and video segmentation. Its central contribution is medical Promptable Concept Segmentation using semantic labels such as anatomical structure names and pathology concepts.

[P00419 | 22413:22531 | NORMAL_TEXT]
It also introduces a MedSAM3 Agent that uses a multimodal language model for reasoning and iterative mask refinement.

[P00420 | 22531:22606 | NORMAL_TEXT]
However, based on the published description, MedSAM3 primarily focuses on:

[P00421 | 22606:22642 | NORMAL_TEXT | LIST id=kix.6h491wubbb6x level=0]
concept-based medical segmentation;

[P00422 | 22642:22674 | NORMAL_TEXT | LIST id=kix.6h491wubbb6x level=0]
open-vocabulary text prompting;

[P00423 | 22674:22701 | NORMAL_TEXT | LIST id=kix.6h491wubbb6x level=0]
medical domain adaptation;

[P00424 | 22701:22736 | NORMAL_TEXT | LIST id=kix.6h491wubbb6x level=0]
agent-guided iterative refinement;

[P00425 | 22736:22778 | NORMAL_TEXT | LIST id=kix.6h491wubbb6x level=0]
generalization across imaging modalities.

[P00426 | 22778:22877 | NORMAL_TEXT]
It does not appear to make reliability-calibrated temporal memory admission its main contribution.

[P00427 | 22877:22969 | NORMAL_TEXT]
I found no clear evidence in the original MedSAM3 paper of a dedicated mechanism combining:

[P00428 | 22969:22995 | NORMAL_TEXT | LIST id=kix.8jie0yh8a50n level=0]
frame-quality estimation;

[P00429 | 22995:23023 | NORMAL_TEXT | LIST id=kix.8jie0yh8a50n level=0]
calibrated mask confidence;

[P00430 | 23023:23046 | NORMAL_TEXT | LIST id=kix.8jie0yh8a50n level=0]
prompt–mask agreement;

[P00431 | 23046:23073 | NORMAL_TEXT | LIST id=kix.8jie0yh8a50n level=0]
temporal drift estimation;

[P00432 | 23073:23118 | NORMAL_TEXT | LIST id=kix.8jie0yh8a50n level=0]
explicit update/freeze/reset memory actions.

[P00433 | 23118:23160 | NORMAL_TEXT]
This leaves a potentially meaningful gap.

[P00434 | 23160:23161 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00435 | 23164:23178 | NORMAL_TEXT | TABLE row=0 col=0]
Research line

[P00436 | 23179:23201 | NORMAL_TEXT | TABLE row=0 col=1]
Representative papers

[P00437 | 23202:23226 | NORMAL_TEXT | TABLE row=0 col=2]
What they already solve

[P00438 | 23227:23249 | NORMAL_TEXT | TABLE row=0 col=3]
Remaining opportunity

[P00439 | 23251:23275 | NORMAL_TEXT | TABLE row=1 col=0]
Adaptive medical memory

[P00440 | 23276:23308 | NORMAL_TEXT | TABLE row=1 col=1]
Medical SAM 2, MedSAM2, SAMed-2

[P00441 | 23309:23365 | NORMAL_TEXT | TABLE row=1 col=2]
Confidence-based memory selection, similarity retrieval

[P00442 | 23366:23431 | NORMAL_TEXT | TABLE row=1 col=3]
Endoscopy-specific memory admission and contamination prevention

[P00443 | 23433:23457 | NORMAL_TEXT | TABLE row=2 col=0]
Polyp-video reliability

[P00444 | 23458:23484 | NORMAL_TEXT | TABLE row=2 col=1]
MICCAI SQA, SALI, ARTEMIS

[P00445 | 23485:23555 | NORMAL_TEXT | TABLE row=2 col=2]
Detect or repair unreliable frames, reconstruct from reliable history

[P00446 | 23556:23639 | NORMAL_TEXT | TABLE row=2 col=3]
Direct control of internal MedSAM2 updates using calibrated multimodal reliability

[P00447 | 23641:23664 | NORMAL_TEXT | TABLE row=3 col=0]
SAM2 memory robustness

[P00448 | 23665:23683 | NORMAL_TEXT | TABLE row=3 col=1]
MA-SAM2, SAM2Long

[P00449 | 23684:23742 | NORMAL_TEXT | TABLE row=3 col=2]
Occlusion-resilient memory, multiple propagation pathways

[P00450 | 23743:23818 | NORMAL_TEXT | TABLE row=3 col=3]
Lightweight single-path gating for deformable tissue and image degradation

[P00451 | 23820:23843 | NORMAL_TEXT | TABLE row=4 col=0]
Interactive correction

[P00452 | 23844:23875 | NORMAL_TEXT | TABLE row=4 col=1]
QDMN/uncertainty-triggered VOS

[P00453 | 23876:23916 | NORMAL_TEXT | TABLE row=4 col=2]
Trigger human correction when uncertain

[P00454 | 23917:23977 | NORMAL_TEXT | TABLE row=4 col=3]
Clinically meaningful prompt/correction-burden optimization

[P00455 | 23979:23998 | NORMAL_TEXT | TABLE row=5 col=0]
Temporal smoothing

[P00456 | 23999:24029 | NORMAL_TEXT | TABLE row=5 col=1]
Probability smoothing methods

[P00457 | 24030:24068 | NORMAL_TEXT | TABLE row=5 col=2]
Reduce flicker and output instability

[P00458 | 24069:24145 | NORMAL_TEXT | TABLE row=5 col=3]
Stop wrong predictions from entering memory rather than only smoothing them

[P00459 | 24146:24150 | HEADING_2]
Gap

[P00460 | 24150:24433 | NORMAL_TEXT]
Existing SAM2/MedSAM2 methods select, weight or reuse memories, but there remains limited work on lightweight, explicitly calibrated, endoscopy-specific admission control that prevents unreliable masks from entering memory and triggers recovery based on estimated future drift risk.

[P00461 | 24433:24435 | NORMAL_TEXT]
[INLINE_OBJECT kix.40x91appana7]

[P00462 | 24435:24465 | HEADING_2]
[16. ReMATF — Liu et al., 2026](https://arxiv.org/pdf/2605.21440)

[P00463 | 24465:24521 | NORMAL_TEXT]
Why it is relevant: This video-restoration method uses:

[P00464 | 24521:24550 | NORMAL_TEXT | LIST id=kix.gwwrevyh2v68 level=0]
a current-frame restoration;

[P00465 | 24550:24580 | NORMAL_TEXT | LIST id=kix.gwwrevyh2v68 level=0]
the previous restored output;

[P00466 | 24580:24598 | NORMAL_TEXT | LIST id=kix.gwwrevyh2v68 level=0]
temporal warping;

[P00467 | 24598:24634 | NORMAL_TEXT | LIST id=kix.gwwrevyh2v68 level=0]
motion-adaptive, pixel-wise fusion.

[P00468 | 24634:24763 | NORMAL_TEXT]
Its module directly fuses the warped previous output with the current prediction while using only two frames at each step.[https://arxiv.org/abs/2605.21440?utm_source=chatgpt.com](https://arxiv.org/abs/2605.21440?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2605.21440?utm_source=chatgpt.com)

[P00469 | 24763:24845 | NORMAL_TEXT]
Overlap with your idea: Extremely high mathematically, although the task differs.

[P00470 | 24845:24880 | NORMAL_TEXT]
Difference from your contribution:

[P00471 | 24880:24928 | NORMAL_TEXT | LIST id=kix.y13poeadalzj level=0]
It performs atmospheric-turbulence restoration.

[P00472 | 24928:24958 | NORMAL_TEXT | LIST id=kix.y13poeadalzj level=0]
Its fusion module is learned.

[P00473 | 24958:25015 | NORMAL_TEXT | LIST id=kix.y13poeadalzj level=0]
Reliability is based on motion and restoration features.

[P00474 | 25015:25117 | NORMAL_TEXT | LIST id=kix.y13poeadalzj level=0]
Your system uses interpretable segmentation-quality signals and prompts MedSAM2 with the fused state.

[P00475 | 25117:25207 | NORMAL_TEXT]
Read for: recurrent feedback and motion-adaptive per-pixel blending outside segmentation.

[P00476 | 25207:25208 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00477 | 25208:25232 | HEADING_2]
Highest-priority papers

[P00478 | 25232:25327 | HEADING_3]
1. Inference-Time Temporal Probability Smoothing for Stable Video Segmentation — 2026 preprint

[P00479 | 25327:25390 | NORMAL_TEXT]
This is the closest paper to your probability-fusion equation.

[P00480 | 25390:25749 | NORMAL_TEXT]
It takes the previous refined probability mask, warps it to the current frame using optical flow, and adaptively combines it with the current SAM2 probability map. Its fusion is pixel-wise and uses segmentation uncertainty together with optical-flow consistency to decide whether each pixel should trust the current prediction or historical prediction.[https://arxiv.org/html/2604.17115v1?utm_source=chatgpt.com](https://arxiv.org/html/2604.17115v1?utm_source=chatgpt.com)[arXiv](https://arxiv.org/html/2604.17115v1?utm_source=chatgpt.com)

[P00481 | 25749:25751 | NORMAL_TEXT]
[INLINE_OBJECT kix.91c2njs8pe1x]

[P00482 | 25751:25776 | NORMAL_TEXT]
Overlap with your method

[P00483 | 25776:25791 | NORMAL_TEXT | LIST id=kix.rg79x2ckfe4b level=0]
training-free;

[P00484 | 25791:25804 | NORMAL_TEXT | LIST id=kix.rg79x2ckfe4b level=0]
frozen SAM2;

[P00485 | 25804:25850 | NORMAL_TEXT | LIST id=kix.rg79x2ckfe4b level=0]
current and previous probability-mask fusion;

[P00486 | 25850:25874 | NORMAL_TEXT | LIST id=kix.rg79x2ckfe4b level=0]
optical-flow alignment;

[P00487 | 25874:25922 | NORMAL_TEXT | LIST id=kix.rg79x2ckfe4b level=0]
reliability or uncertainty-dependent weighting;

[P00488 | 25922:25948 | NORMAL_TEXT | LIST id=kix.rg79x2ckfe4b level=0]
constant recurrent state.

[P00489 | 25948:25969 | NORMAL_TEXT]
Important difference

[P00490 | 25969:26215 | NORMAL_TEXT]
The method performs output-level temporal smoothing. Your architecture feeds the fused state back as a dense mask prompt for the next frame. Therefore, your method affects future prediction, while this paper mainly stabilizes the current output.

[P00491 | 26215:26295 | NORMAL_TEXT]
Read this first. It directly exposes the duplicate-mask problem you identified.

[P00492 | 26295:26297 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00493 | 26297:26399 | HEADING_3]
2. SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree — ICCV 2025

[P00494 | 26399:26781 | NORMAL_TEXT]
SAM2Long addresses error accumulation caused by SAM2’s greedy memory propagation. Rather than committing to one predicted mask at every frame, it maintains several possible propagation pathways. Candidate branches are ranked using cumulative confidence, and weak branches are discarded. This makes it more robust to occlusion, object disappearance and reappearance.[https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html?utm_source=chatgpt.com](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html?utm_source=chatgpt.com)[Open Access CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html?utm_source=chatgpt.com)

[P00495 | 26781:26789 | NORMAL_TEXT]
Overlap

[P00496 | 26789:26804 | NORMAL_TEXT | LIST id=kix.bx2ilio4n4zk level=0]
training-free;

[P00497 | 26804:26817 | NORMAL_TEXT | LIST id=kix.bx2ilio4n4zk level=0]
frozen SAM2;

[P00498 | 26817:26864 | NORMAL_TEXT | LIST id=kix.bx2ilio4n4zk level=0]
explicitly targets erroneous-mask propagation;

[P00499 | 26864:26943 | NORMAL_TEXT | LIST id=kix.bx2ilio4n4zk level=0]
uses uncertainty to prevent early mistakes from controlling all future frames;

[P00500 | 26943:26983 | NORMAL_TEXT | LIST id=kix.bx2ilio4n4zk level=0]
handles disappearance and reappearance.

[P00501 | 26983:26994 | NORMAL_TEXT]
Difference

[P00502 | 26994:26996 | NORMAL_TEXT]
[INLINE_OBJECT kix.hlrx0gvwax7y]

[P00503 | 26996:26998 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00504 | 26998:27076 | HEADING_3]
3. A Distractor-Aware Memory for Visual Object Tracking with SAM2 — CVPR 2025

[P00505 | 27076:27363 | NORMAL_TEXT]
This paper introduces DAM4SAM, which uses a distractor-aware memory and an introspection-based update strategy. It examines whether a new observation is useful and safe before updating the target representation, particularly when similar-looking distractors are present.[https://openaccess.thecvf.com/content/CVPR2025/html/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.html?utm_source=chatgpt.com](https://openaccess.thecvf.com/content/CVPR2025/html/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.html?utm_source=chatgpt.com)[Open Access CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.html?utm_source=chatgpt.com)

[P00506 | 27363:27371 | NORMAL_TEXT]
Overlap

[P00507 | 27371:27396 | NORMAL_TEXT | LIST id=kix.bkuhqqscu12 level=0]
selective memory update;

[P00508 | 27396:27458 | NORMAL_TEXT | LIST id=kix.bkuhqqscu12 level=0]
distinguishes the true tracked object from competing regions;

[P00509 | 27458:27518 | NORMAL_TEXT | LIST id=kix.bkuhqqscu12 level=0]
avoids blindly placing every recent prediction into memory;

[P00510 | 27518:27580 | NORMAL_TEXT | LIST id=kix.bkuhqqscu12 level=0]
relevant to your duplicate-object and wrong-location concern.

[P00511 | 27580:27591 | NORMAL_TEXT]
Difference

[P00512 | 27591:27801 | NORMAL_TEXT]
It targets visual object tracking and distractor discrimination rather than explicit probability-mask averaging. Nevertheless, its memory-admission logic is highly relevant to your proposed reliability scorer.

[P00513 | 27801:27803 | NORMAL_TEXT]
[INLINE_OBJECT kix.9e0igjy9nnqj]

[P00514 | 27803:27865 | NORMAL_TEXT]
A prediction may be confident but correspond to a distractor.

[P00515 | 27865:27867 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00516 | 27867:27963 | HEADING_3]
4. MA-SAM2: Memory-Augmented SAM2 for Training-Free Surgical Video Segmentation — 2025 preprint

[P00517 | 27963:28202 | NORMAL_TEXT]
MA-SAM2 proposes training-free, context-aware and occlusion-resilient memory mechanisms for surgical videos. It directly studies instability caused by uncertain intermediate masks, instrument disappearance, reappearance and overlap.[https://arxiv.org/abs/2507.09577?utm_source=chatgpt.com](https://arxiv.org/abs/2507.09577?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2507.09577?utm_source=chatgpt.com)

[P00518 | 28202:28210 | NORMAL_TEXT]
Overlap

[P00519 | 28210:28225 | NORMAL_TEXT | LIST id=kix.yn5sveli5ub8 level=0]
training-free;

[P00520 | 28225:28238 | NORMAL_TEXT | LIST id=kix.yn5sveli5ub8 level=0]
frozen SAM2;

[P00521 | 28238:28261 | NORMAL_TEXT | LIST id=kix.yn5sveli5ub8 level=0]
medical-video setting;

[P00522 | 28261:28289 | NORMAL_TEXT | LIST id=kix.yn5sveli5ub8 level=0]
occlusion and reappearance;

[P00523 | 28289:28326 | NORMAL_TEXT | LIST id=kix.yn5sveli5ub8 level=0]
unreliable intermediate predictions;

[P00524 | 28326:28370 | NORMAL_TEXT | LIST id=kix.yn5sveli5ub8 level=0]
memory design rather than model retraining.

[P00525 | 28370:28381 | NORMAL_TEXT]
Difference

[P00526 | 28381:28565 | NORMAL_TEXT]
It modifies or augments memory selection using richer contextual histories, whereas your design eliminates the normal FIFO memory and maintains one externally fused probability state.

[P00527 | 28565:28653 | NORMAL_TEXT]
This is one of the strongest architectural competitors for your healthcare application.

[P00528 | 28653:28655 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00529 | 28655:28689 | HEADING_2]
Highly relevant supporting papers

[P00530 | 28689:28770 | HEADING_3]
5. LiVOS: Light Video Object Segmentation with Gated Linear Matching — CVPR 2025

[P00531 | 28770:28954 | NORMAL_TEXT]
LiVOS replaces an expanding space-time memory with a recurrent constant-size state. A learned data-dependent gate controls which information is retained or discarded.[https://openaccess.thecvf.com/content/CVPR2025/html/Liu_LiVOS_Light_Video_Object_Segmentation_with_Gated_Linear_Matching_CVPR_2025_paper.html?utm_source=chatgpt.com](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_LiVOS_Light_Video_Object_Segmentation_with_Gated_Linear_Matching_CVPR_2025_paper.html?utm_source=chatgpt.com)[Open Access CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_LiVOS_Light_Video_Object_Segmentation_with_Gated_Linear_Matching_CVPR_2025_paper.html?utm_source=chatgpt.com)[INLINE_OBJECT kix.gwd893pvjzcf]

[P00532 | 28954:28962 | NORMAL_TEXT]
Overlap

[P00533 | 28962:28998 | NORMAL_TEXT | LIST id=kix.8nkm1gvmneuf level=0]
constant-capacity recurrent memory;

[P00534 | 28998:29028 | NORMAL_TEXT | LIST id=kix.8nkm1gvmneuf level=0]
explicit retain/discard gate;

[P00535 | 29028:29067 | NORMAL_TEXT | LIST id=kix.8nkm1gvmneuf level=0]
avoids storing every historical frame;

[P00536 | 29067:29124 | NORMAL_TEXT | LIST id=kix.8nkm1gvmneuf level=0]
very close to your “one updated memory \(M_t\)” concept.

[P00537 | 29124:29135 | NORMAL_TEXT]
Difference

[P00538 | 29135:29284 | NORMAL_TEXT]
LiVOS uses a learned feature-level gate and requires training. Your method applies heuristic reliability at probability-mask level without training.

[P00539 | 29284:29333 | NORMAL_TEXT]
This paper is important for framing your method:

[P00540 | 29333:29439 | NORMAL_TEXT]
Your external mask state can be viewed as a training-free, mask-space analogue of gated recurrent memory.

[P00541 | 29439:29441 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00542 | 29441:29554 | HEADING_3]
6. SAM-I2V: Upgrading SAM to Support Promptable Video Segmentation with Less Than 0.2% Training Cost — CVPR 2025

[P00543 | 29554:29574 | NORMAL_TEXT]
SAM-I2V introduces:

[P00544 | 29574:29637 | NORMAL_TEXT | LIST id=kix.b6peeacfuvsv level=0]
a memory-filtering strategy that selects relevant past frames;

[P00545 | 29637:29730 | NORMAL_TEXT | LIST id=kix.b6peeacfuvsv level=0]
memory-as-prompt, where historical object information is used to guide current segmentation;

[P00546 | 29730:29808 | NORMAL_TEXT | LIST id=kix.b6peeacfuvsv level=0]
temporal propagation using selected rather than indiscriminate history.[https://arxiv.org/html/2506.01304v1?utm_source=chatgpt.com](https://arxiv.org/html/2506.01304v1?utm_source=chatgpt.com)[arXiv](https://arxiv.org/html/2506.01304v1?utm_source=chatgpt.com)

[P00547 | 29808:29816 | NORMAL_TEXT]
Overlap

[P00548 | 29816:29841 | NORMAL_TEXT | LIST id=kix.v8v5svy0s4cy level=0]
memory used as a prompt;

[P00549 | 29841:29875 | NORMAL_TEXT | LIST id=kix.v8v5svy0s4cy level=0]
selective historical information;

[P00550 | 29875:29915 | NORMAL_TEXT | LIST id=kix.v8v5svy0s4cy level=0]
temporally consistent mask propagation;

[P00551 | 29915:29973 | NORMAL_TEXT | LIST id=kix.v8v5svy0s4cy level=0]
particularly relevant to your dense mask-prompt feedback.

[P00552 | 29973:29984 | NORMAL_TEXT]
Difference

[P00553 | 29984:30153 | NORMAL_TEXT]
It trains additional video components and retains several historical memories. Your approach is training-free and compresses history into one external probability mask.

[P00554 | 30153:30155 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00555 | 30155:30249 | HEADING_3]
7. Structure Matters: Revisiting Boundary Refinement in Video Object Segmentation — ICCV 2025

[P00556 | 30249:30445 | NORMAL_TEXT]
OASIS combines stored object features with structural edge information and introduces evidential uncertainty estimation, particularly for occluded regions and object interactions.[https://openaccess.thecvf.com/content/ICCV2025/html/Qin_Structure_Matters_Revisiting_Boundary_Refinement_in_Video_Object_Segmentation_ICCV_2025_paper.html?utm_source=chatgpt.com](https://openaccess.thecvf.com/content/ICCV2025/html/Qin_Structure_Matters_Revisiting_Boundary_Refinement_in_Video_Object_Segmentation_ICCV_2025_paper.html?utm_source=chatgpt.com)[Open Access CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Qin_Structure_Matters_Revisiting_Boundary_Refinement_in_Video_Object_Segmentation_ICCV_2025_paper.html?utm_source=chatgpt.com)

[P00557 | 30445:30453 | NORMAL_TEXT]
Overlap

[P00558 | 30453:30495 | NORMAL_TEXT | LIST id=kix.w9emnai5d7jz level=0]
boundary quality as a reliability signal;

[P00559 | 30495:30528 | NORMAL_TEXT | LIST id=kix.w9emnai5d7jz level=0]
explicit uncertainty estimation;

[P00560 | 30528:30556 | NORMAL_TEXT | LIST id=kix.w9emnai5d7jz level=0]
occlusion-aware refinement;

[P00561 | 30556:30616 | NORMAL_TEXT | LIST id=kix.w9emnai5d7jz level=0]
prevents poor object boundaries from being trusted blindly.

[P00562 | 30616:30627 | NORMAL_TEXT]
Difference

[P00563 | 30627:30810 | NORMAL_TEXT]
It learns a refinement network and works mainly at feature and boundary-representation level. Your proposed reliability scorer uses training-free boundary plausibility as one input.[INLINE_OBJECT kix.dm24cr1lq260]

[P00564 | 30810:30935 | NORMAL_TEXT]
in your reliability score, but it also shows that boundary confidence should ideally be spatial rather than a single scalar.

[P00565 | 30935:30937 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00566 | 30937:31022 | HEADING_3]
8. SAMed-2: Selective Memory Enhanced Medical Segment Anything Model — 2025 preprint

[P00567 | 31022:31228 | NORMAL_TEXT]
SAMed-2 proposes a confidence-driven memory mechanism that stores high-confidence features and retrieves memories using similarity. Its goal is to reduce noise and forgetting in medical segmentation.[https://arxiv.org/html/2507.03698v1?utm_source=chatgpt.com](https://arxiv.org/html/2507.03698v1?utm_source=chatgpt.com)[arXiv](https://arxiv.org/html/2507.03698v1?utm_source=chatgpt.com)

[P00568 | 31228:31236 | NORMAL_TEXT]
Overlap

[P00569 | 31236:31258 | NORMAL_TEXT | LIST id=kix.1j34w7f63zoy level=0]
medical segmentation;

[P00570 | 31258:31293 | NORMAL_TEXT | LIST id=kix.1j34w7f63zoy level=0]
confidence-based memory admission;

[P00571 | 31293:31312 | NORMAL_TEXT | LIST id=kix.1j34w7f63zoy level=0]
selective storage;

[P00572 | 31312:31389 | NORMAL_TEXT | LIST id=kix.1j34w7f63zoy level=0]
explicitly avoids allowing low-confidence information to contaminate memory.

[P00573 | 31389:31400 | NORMAL_TEXT]
Difference

[P00574 | 31400:31581 | NORMAL_TEXT]
The confidence mechanism is learned during training, and selection occurs in feature space. Your method is inference-only and uses probability masks plus heuristic quality signals.

[P00575 | 31581:31713 | NORMAL_TEXT]
This is important because you should not claim that confidence-guided medical memory is entirely new. Your stronger distinction is:

[P00576 | 31713:31793 | NORMAL_TEXT]
training-free external recurrent mask-state control for a frozen MedSAM2 model.

[P00577 | 31793:31795 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00578 | 31795:31924 | HEADING_3]
9. Find First, Track Next: Decoupling Identification and Propagation in Referring Video Object Segmentation — ICCV Workshop 2025

[P00579 | 31924:32148 | NORMAL_TEXT]
FindTrack selects a reliable keyframe by balancing segmentation confidence with vision-language alignment, then uses that reference for propagation. It separates object identification from temporal tracking.[https://openaccess.thecvf.com/content/ICCV2025W/LSVOS/html/Cho_Find_First_Track_Next_Decoupling_Identification_and_Propagation_in_Referring_ICCVW_2025_paper.html?utm_source=chatgpt.com](https://openaccess.thecvf.com/content/ICCV2025W/LSVOS/html/Cho_Find_First_Track_Next_Decoupling_Identification_and_Propagation_in_Referring_ICCVW_2025_paper.html?utm_source=chatgpt.com)[Open Access CVF](https://openaccess.thecvf.com/content/ICCV2025W/LSVOS/html/Cho_Find_First_Track_Next_Decoupling_Identification_and_Propagation_in_Referring_ICCVW_2025_paper.html?utm_source=chatgpt.com)

[P00580 | 32148:32156 | NORMAL_TEXT]
Overlap

[P00581 | 32156:32186 | NORMAL_TEXT | LIST id=kix.x3xq6vvppbvv level=0]
reliable reference selection;

[P00582 | 32186:32206 | NORMAL_TEXT | LIST id=kix.x3xq6vvppbvv level=0]
prompt reliability;

[P00583 | 32206:32253 | NORMAL_TEXT | LIST id=kix.x3xq6vvppbvv level=0]
avoids beginning propagation from a poor mask;

[P00584 | 32253:32310 | NORMAL_TEXT | LIST id=kix.x3xq6vvppbvv level=0]
useful for text-prompted adenoid or airway segmentation.

[P00585 | 32310:32321 | NORMAL_TEXT]
Difference

[P00586 | 32321:32437 | NORMAL_TEXT]
It mainly decides where propagation should start, whereas your method decides what to do at every recurrent update.

[P00587 | 32437:32507 | NORMAL_TEXT]
For MedSAM3 or text-prompt initialization, this is especially useful.

[P00588 | 32507:32509 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00589 | 32509:32627 | HEADING_3]
10. MPG-SAM 2: Adapting SAM 2 with Mask Priors and Global Context for Referring Video Object Segmentation — ICCV 2025

[P00590 | 32627:32830 | NORMAL_TEXT]
MPG-SAM2 generates pseudo-mask priors and supplies them to SAM2 as dense prompts. It combines global video information with historical object information to improve temporal consistency.[https://openaccess.thecvf.com/content/ICCV2025/html/Rong_MPG-SAM_2_Adapting_SAM_2_with_Mask_Priors_and_Global_ICCV_2025_paper.html?utm_source=chatgpt.com](https://openaccess.thecvf.com/content/ICCV2025/html/Rong_MPG-SAM_2_Adapting_SAM_2_with_Mask_Priors_and_Global_ICCV_2025_paper.html?utm_source=chatgpt.com)[Open Access CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Rong_MPG-SAM_2_Adapting_SAM_2_with_Mask_Priors_and_Global_ICCV_2025_paper.html?utm_source=chatgpt.com)

[P00591 | 32830:32838 | NORMAL_TEXT]
Overlap

[P00592 | 32838:32892 | NORMAL_TEXT | LIST id=kix.i82bvaox3ary level=0]
dense mask prior supplied through the prompt encoder;

[P00593 | 32892:32921 | NORMAL_TEXT | LIST id=kix.i82bvaox3ary level=0]
historical mask information;

[P00594 | 32921:32979 | NORMAL_TEXT | LIST id=kix.i82bvaox3ary level=0]
current prediction guided by an external mask-like state;

[P00595 | 32979:33011 | NORMAL_TEXT | LIST id=kix.i82bvaox3ary level=0]
text-guided video segmentation.

[P00596 | 33011:33022 | NORMAL_TEXT]
Difference

[P00597 | 33022:33143 | NORMAL_TEXT]
It requires training and uses global offline context. Your design is causal, training-free and uses one recurrent state.

[P00598 | 33143:33184 | HEADING_2]
Best reading order for your architecture

[P00599 | 33184:33261 | NORMAL_TEXT | LIST id=kix.h71001s7va4 level=0]
Inference-Time Temporal Probability Smoothing — exact mask-fusion mechanism.

[P00600 | 33261:33319 | NORMAL_TEXT | LIST id=kix.h71001s7va4 level=0]
SAM2Long — uncertainty and alternative propagation paths.

[P00601 | 33319:33370 | NORMAL_TEXT | LIST id=kix.h71001s7va4 level=0]
DAM4SAM — introspection and safe memory admission.

[P00602 | 33370:33427 | NORMAL_TEXT | LIST id=kix.h71001s7va4 level=0]
MA-SAM2 — training-free medical-video memory robustness.

[P00603 | 33427:33464 | NORMAL_TEXT | LIST id=kix.h71001s7va4 level=0]
LiVOS — constant-state gated memory.

[P00604 | 33464:33492 | NORMAL_TEXT | LIST id=kix.h71001s7va4 level=0]
SAM-I2V — memory-as-prompt.

[P00605 | 33492:33536 | NORMAL_TEXT | LIST id=kix.h71001s7va4 level=0]
SAMed-2 — confidence-driven medical memory.

[P00606 | 33536:33578 | NORMAL_TEXT | LIST id=kix.h71001s7va4 level=0]
OASIS — boundary and spatial uncertainty.

[P00607 | 33578:33636 | HEADING_2]
What these papers imply for your probability-fusion block

[P00608 | 33636:33706 | NORMAL_TEXT]
Across this literature, three distinct operations repeatedly appear:[INLINE_OBJECT kix.8pk89iyo4nq1]

[P00609 | 33706:33851 | NORMAL_TEXT]
is no longer sufficient as the full method. The more defensible contribution is an agreement-aware state-transition policy that determines both:

[P00610 | 33851:33875 | NORMAL_TEXT | LIST id=kix.7o4krlalyij3 level=0]
current output decision

[P00611 | 33875:33906 | NORMAL_TEXT | LIST id=kix.7o4krlalyij3 level=0]
next-memory admission decision

[P00612 | 33906:33990 | NORMAL_TEXT]
That distinction remains promising and is not identical to the closest papers above

[P00613 | 33990:34044 | HEADING_3]
1. Choo et al. — Motion–appearance probability fusion

[P00614 | 34044:34148 | NORMAL_TEXT]
Automatic Video Object Segmentation via Motion-Appearance-Stream Fusion and Instance-aware Segmentation

[P00615 | 34148:34223 | NORMAL_TEXT]
This is one of the clearest precedents for your idea. The method produces:

[P00616 | 34223:34295 | NORMAL_TEXT | LIST id=kix.5eje8hlerd3t level=0]
a pixel-level foreground probability map from motion–appearance fusion;

[P00617 | 34295:34351 | NORMAL_TEXT | LIST id=kix.5eje8hlerd3t level=0]
an instance segmentation mask with an objectness score;

[P00618 | 34351:34448 | NORMAL_TEXT | LIST id=kix.5eje8hlerd3t level=0]
a final result obtained by combining the foreground probabilities and instance-level confidence.

[P00619 | 34448:34721 | NORMAL_TEXT]
The paper reports that this fusion achieves state-of-the-art automatic VOS performance at the time and approaches semi-supervised performance. It supports the argument that two imperfect but complementary mask estimates can become stronger when fused at probability level.

[P00620 | 34721:34742 | NORMAL_TEXT]
Connection to AdeSEG

[P00621 | 34742:34744 | NORMAL_TEXT]
[INLINE_OBJECT kix.wcftdfncandl]

[P00622 | 34744:34746 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00623 | 34746:34787 | HEADING_3]
2. Sun et al. — Mask Propagation Network

[P00624 | 34787:34842 | NORMAL_TEXT]
Mask Propagation Network for Video Object Segmentation

[P00625 | 34842:34870 | NORMAL_TEXT]
This paper explicitly uses:

[P00626 | 34870:34907 | NORMAL_TEXT | LIST id=kix.s8qvaisvjvvq level=0]
the previous-frame mask probability;

[P00627 | 34907:34947 | NORMAL_TEXT | LIST id=kix.s8qvaisvjvvq level=0]
an optical-flow-warped probability map;

[P00628 | 34947:34973 | NORMAL_TEXT | LIST id=kix.s8qvaisvjvvq level=0]
current-frame appearance;

[P00629 | 34973:35064 | NORMAL_TEXT | LIST id=kix.s8qvaisvjvvq level=0]
an additional segmentation model whose output is ensembled with the propagated prediction.

[P00630 | 35064:35285 | NORMAL_TEXT]
This is strong support for preserving the mask as a probability map during temporal propagation rather than immediately binarizing it. The ensemble component is also intended to recover objects that propagation may lose.

[P00631 | 35285:35315 | NORMAL_TEXT]
Why it supports your proposal

[P00632 | 35315:35359 | NORMAL_TEXT]
Your two branches could play similar roles:

[P00633 | 35359:35403 | NORMAL_TEXT | LIST id=kix.ts4c0cokpxhn level=0]
MedSAM2 memory branch: temporal continuity;

[P00634 | 35403:35458 | NORMAL_TEXT | LIST id=kix.ts4c0cokpxhn level=0]
current-prompt branch: object recovery and correction;

[P00635 | 35458:35521 | NORMAL_TEXT | LIST id=kix.ts4c0cokpxhn level=0]
probability fusion: reconciliation before binary thresholding.

[P00636 | 35521:35523 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00637 | 35523:35587 | HEADING_3]
3. Li et al. — Video Object Segmentation with Re-identification

[P00638 | 35587:35636 | NORMAL_TEXT]
Video Object Segmentation with Re-identification

[P00639 | 35636:35657 | NORMAL_TEXT]
The method combines:

[P00640 | 35657:35724 | NORMAL_TEXT | LIST id=kix.1y65m1g2ihf level=0]
a mask-propagation module producing a flow-warped probability map;

[P00641 | 35724:35835 | NORMAL_TEXT | LIST id=kix.1y65m1g2ihf level=0]
a re-identification module that recovers the target after large motion, disappearance, or propagation failure.

[P00642 | 35835:35986 | NORMAL_TEXT]
The paper was explicitly motivated by drift and failure under large displacement. It achieved a global DAVIS J&F score of 0.699 in the 2017 challenge.

[P00643 | 35986:36032 | NORMAL_TEXT]
This supports a central point in your thesis:

[P00644 | 36032:36184 | NORMAL_TEXT]
Temporal propagation should not be trusted alone; it should be combined with an independent current-frame observation capable of recovering from drift.

[P00645 | 36184:36186 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00646 | 36186:36234 | HEADING_3]
4. Cheng et al. — MiVOS difference-aware fusion

[P00647 | 36234:36342 | NORMAL_TEXT]
Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion

[P00648 | 36342:36359 | NORMAL_TEXT]
MiVOS separates:

[P00649 | 36359:36394 | NORMAL_TEXT | LIST id=kix.yl8489drofsv level=0]
interaction-based mask prediction;

[P00650 | 36394:36421 | NORMAL_TEXT | LIST id=kix.yl8489drofsv level=0]
temporal mask propagation;

[P00651 | 36421:36446 | NORMAL_TEXT | LIST id=kix.yl8489drofsv level=0]
difference-aware fusion.

[P00652 | 36446:36705 | NORMAL_TEXT]
Its fusion module learns how to combine masks produced before and after a corrective interaction. The masks are aligned to target frames through space-time memory before fusion. MiVOS outperformed contemporary methods while requiring fewer user interactions.

[P00653 | 36705:36759 | NORMAL_TEXT]
This is particularly close to your clinical use case:

[P00654 | 36759:36823 | NORMAL_TEXT | LIST id=kix.a1ptkpsszia8 level=0]
the propagated MedSAM2 mask represents the existing trajectory;

[P00655 | 36823:36882 | NORMAL_TEXT | LIST id=kix.a1ptkpsszia8 level=0]
a detector or clinician prompt produces a corrective mask;

[P00656 | 36882:36983 | NORMAL_TEXT | LIST id=kix.a1ptkpsszia8 level=0]
the system fuses the old and corrected evidence rather than completely replacing one with the other.

[P00657 | 36983:37078 | NORMAL_TEXT]
This paper can support both probability fusion and your intended correction-burden evaluation.

[P00658 | 37078:37080 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00659 | 37080:37133 | HEADING_3]
5. STMask — temporal fusion handles difficult frames

[P00660 | 37133:37256 | NORMAL_TEXT]
Spatial Feature Calibration and Temporal Fusion for Effective One-stage Video Instance Segmentation Li et al., CVPR 2021.

[P00661 | 37256:37549 | NORMAL_TEXT]
STMask introduces temporal fusion between adjacent frames and reports improved video instance segmentation performance, particularly under motion blur, partial occlusion, and unusual object–camera poses. It achieved mask AP values of 33.5 and 36.8 with ResNet-50 and ResNet-101, respectively.

[P00662 | 37549:37728 | NORMAL_TEXT]
This supports the broader claim that temporal information should be fused with current-frame evidence, especially for precisely the difficult conditions encountered in endoscopy:

[P00663 | 37728:37734 | NORMAL_TEXT | LIST id=kix.a3hrjlor2zs7 level=0]
blur;

[P00664 | 37734:37745 | NORMAL_TEXT | LIST id=kix.a3hrjlor2zs7 level=0]
occlusion;

[P00665 | 37745:37766 | NORMAL_TEXT | LIST id=kix.a3hrjlor2zs7 level=0]
rapid camera motion;

[P00666 | 37766:37786 | NORMAL_TEXT | LIST id=kix.a3hrjlor2zs7 level=0]
changing viewpoint.

[P00667 | 37786:37943 | NORMAL_TEXT]
However, STMask fuses features rather than final probability masks, so it is best cited as supporting evidence rather than an exact architectural precedent.

[P00668 | 37943:37984 | HEADING_2]
Evidence from medical image segmentation

[P00669 | 37984:38025 | HEADING_3]
6. SoftSeg — soft masks can improve Dice

[P00670 | 38025:38143 | NORMAL_TEXT]
SoftSeg: Advantages of Soft versus Binary Training for Image Segmentation Gros et al., Medical Image Analysis, 2021.

[P00671 | 38143:38323 | NORMAL_TEXT]
SoftSeg directly studies the use of continuous soft segmentation representations rather than forcing binary outputs during learning. It reports Dice improvements of approximately:

[P00672 | 38323:38362 | NORMAL_TEXT | LIST id=kix.atvk20dp57h1 level=0]
2.0 percentage points for gray matter;

[P00673 | 38362:38392 | NORMAL_TEXT | LIST id=kix.atvk20dp57h1 level=0]
3.3 points for brain lesions;

[P00674 | 38392:38442 | NORMAL_TEXT | LIST id=kix.atvk20dp57h1 level=0]
improvements on spinal cord segmentation as well.

[P00675 | 38442:38571 | NORMAL_TEXT]
This paper does not perform video fusion, but it gives strong medical-imaging support for your underlying representation choice:

[P00676 | 38571:38686 | NORMAL_TEXT]
Probability-valued masks preserve information about ambiguity and boundaries that is lost after hard thresholding.

[P00677 | 38686:38831 | NORMAL_TEXT]
This is highly relevant to adenoid and airway boundaries, which may be ambiguous because of mucus, blur, partial visibility, or reflected light.

[P00678 | 38831:38833 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00679 | 38833:38884 | HEADING_3]
7. Mehrtash et al. — ensembles improve calibration

[P00680 | 38884:39043 | NORMAL_TEXT]
Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation Mehrtash et al., IEEE Transactions on Medical Imaging, 2020.

[P00681 | 39043:39239 | NORMAL_TEXT]
Across brain, cardiac, and prostate segmentation applications, the study shows that model ensembling consistently improves confidence calibration and provides more reliable uncertainty estimates.

[P00682 | 39239:39494 | NORMAL_TEXT]
This supports using multiple probability estimates rather than a single model output. More importantly, it suggests that your fusion should be based on calibrated probabilities, because raw SAM2 logits or confidence scores may not be directly comparable.

[P00683 | 39494:39537 | NORMAL_TEXT]
A stronger formulation would therefore be:

[P00684 | 39537:39577 | NORMAL_TEXT]
Ptfused​=wtp​+wtm​+ϵwtp​Ptp​+wtm​Ptm​​,

[P00685 | 39577:39619 | NORMAL_TEXT]
where P denotes calibrated probabilities.

[P00686 | 39619:39621 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00687 | 39621:39660 | HEADING_3]
8. Uncertainty-aware evidential fusion

[P00688 | 39660:39758 | NORMAL_TEXT]
Uncertainty-aware Evidential Fusion-based Learning for Semi-supervised Medical Image Segmentation

[P00689 | 39758:40069 | NORMAL_TEXT]
This work combines predictions through evidential uncertainty modeling and reports state-of-the-art results on its semi-supervised medical segmentation benchmarks. Its main relevance is that it does not treat every prediction source as equally trustworthy; evidence is fused according to estimated uncertainty.

[P00690 | 40069:40132 | NORMAL_TEXT]
This strengthens a more defensible version of your hypothesis:

[P00691 | 40132:40293 | NORMAL_TEXT]
Performance improvement is not expected from naïve averaging alone, but from reliability-weighted fusion that suppresses uncertain or contradictory predictions.

[P00692 | 40293:40295 | NORMAL_TEXT]
[HORIZONTAL_RULE]

[P00693 | 40295:40342 | HEADING_3]
9. Robust Fusion for Bayesian Semantic Mapping

[P00694 | 40342:40386 | NORMAL_TEXT]
Robust Fusion for Bayesian Semantic Mapping

[P00695 | 40386:40669 | NORMAL_TEXT]
Although this is semantic mapping rather than medical video segmentation, it provides useful theoretical and empirical evidence. The authors find that naïvely accumulating neural-network probabilities can be harmed by overconfident outliers. They improve segmentation robustness by:

[P00696 | 40669:40709 | NORMAL_TEXT | LIST id=kix.6uiiuj5eeail level=0]
regularizing probability distributions;

[P00697 | 40709:40761 | NORMAL_TEXT | LIST id=kix.6uiiuj5eeail level=0]
weighting observations using epistemic uncertainty;

[P00698 | 40761:40811 | NORMAL_TEXT | LIST id=kix.6uiiuj5eeail level=0]
reducing the influence of unreliable predictions.

[P00699 | 40811:40920 | NORMAL_TEXT]
This paper is important because it warns against a weak formulation of your proposal. Simple fusion such as[INLINE_OBJECT kix.88oprafb8rgc]

[P00700 | 40920:40986 | NORMAL_TEXT]
may actually reinforce errors if one branch is confidently wrong.

[P00701 | 40986:41013 | HEADING_2]
Very close recent evidence

[P00702 | 41013:41063 | HEADING_3]
10. Inference-Time Temporal Probability Smoothing

[P00703 | 41063:41157 | NORMAL_TEXT]
Inference-Time Temporal Probability Smoothing for Stable SAM2-Based Video Segmentation 2026.

[P00704 | 41157:41234 | NORMAL_TEXT]
This work operates directly on SAM2 per-frame probability maps and combines:

[P00705 | 41234:41276 | NORMAL_TEXT | LIST id=kix.6l193q5hw4ie level=0]
current-frame segmentation probabilities;

[P00706 | 41276:41320 | NORMAL_TEXT | LIST id=kix.6l193q5hw4ie level=0]
optical-flow-warped temporal probabilities;

[P00707 | 41320:41353 | NORMAL_TEXT | LIST id=kix.6l193q5hw4ie level=0]
entropy-based pixel uncertainty;

[P00708 | 41353:41388 | NORMAL_TEXT | LIST id=kix.6l193q5hw4ie level=0]
forward–backward flow consistency.

[P00709 | 41388:41481 | NORMAL_TEXT]
It reports improved temporal stability without retraining or changing the base architecture.

[P00710 | 41481:41753 | NORMAL_TEXT]
Conceptually, this is probably the closest paper to your intended implementation. However, because it is a very recent 2026 preprint, it should not be your only supporting citation. Use the older VOS and medical-imaging literature above to establish historical grounding.

[P00711 | 41753:41792 | HEADING_2]
What the literature genuinely supports

[P00712 | 41792:41838 | NORMAL_TEXT]
The papers collectively support these claims:

[P00713 | 41838:41946 | NORMAL_TEXT | LIST id=kix.y1kedkp38npq level=0]
Soft masks retain useful uncertainty information. Binarization discards confidence and boundary ambiguity.

[P00714 | 41946:42139 | NORMAL_TEXT | LIST id=kix.y1kedkp38npq level=0]
Temporal propagation and current-frame segmentation are complementary. Propagation improves consistency, while independent frame evidence can recover from drift, disappearance, and occlusion.

[P00715 | 42139:42327 | NORMAL_TEXT | LIST id=kix.y1kedkp38npq level=0]
Fusion can improve segmentation performance. This has been demonstrated through probability-map fusion, temporal fusion, difference-aware fusion, model ensembling, and evidential fusion.

[P00716 | 42327:42523 | NORMAL_TEXT | LIST id=kix.y1kedkp38npq level=0]
Adaptive weighting is safer than fixed averaging. Uncertainty, entropy, temporal consistency, flow reliability, prompt quality, and objectness can determine how much each mask should contribute.

[P00717 | 42523:42722 | NORMAL_TEXT | LIST id=kix.y1kedkp38npq level=0]
Fusion must happen before thresholding. Combining two binary masks by union, intersection, or majority voting loses substantially more information than combining calibrated logits or probabilities.

[P00718 | 42722:42754 | HEADING_2]
A defensible research statement

[P00719 | 42754:42788 | NORMAL_TEXT]
You can frame your hypothesis as:

[P00720 | 42788:43516 | NORMAL_TEXT]
Existing video object segmentation and medical image segmentation studies show that propagated temporal predictions and current-frame observations provide complementary information. Temporal estimates support continuity but may accumulate drift, whereas prompt-conditioned estimates can recover object appearance but may be unstable under poor prompts or image degradation. Motivated by probability-map propagation, difference-aware fusion, soft segmentation, and uncertainty-weighted ensemble methods, we hypothesize that reliability-aware fusion of MedSAM2 memory probabilities and current prompt-conditioned probabilities can improve frame-level accuracy and temporal robustness compared with either prediction source alone.

## benchmark paper sources (t.suey3u60cwn9)

[P00721 | 1:36 | HEADING_2]
Directly relevant benchmark papers

[P00722 | 36:323 | NORMAL_TEXT | LIST id=kix.9bewq29w8lvv level=0]
Self-Prompting Polyp Segmentation in Colonoscopy Using SAM 2 and YOLOv8Mansoori et al., 2024.This is the most important source for your comparison because it explicitly reports results on the 23 PolypGen video sequences. Its Table 3 reports YOLO-SAM 2 with mDice 0.808 and mIoU 0.678.

[P00723 | 323:667 | NORMAL_TEXT | LIST id=kix.9bewq29w8lvv level=0]
HAT-SAM3: Endoscopy-Aware Adaptation of a Foundation Segmentation Model for Generalizable Polyp SegmentationLi et al., 2026.This paper evaluates cross-dataset generalization on PolypGen using a SAM 3-based method. It is useful for a stricter zero-shot or cross-dataset comparison, but its protocol is not necessarily identical to YOLO-SAM 2.

[P00724 | 667:1002 | NORMAL_TEXT | LIST id=kix.9bewq29w8lvv level=0]
DepthPolyp: Pseudo-Depth Guided Lightweight Segmentation for Real-Time ColonoscopyWu et al., 2026.This paper evaluates real surgical video performance on PolypGen. It is especially useful for lightweight and real-time comparisons, although it appears to use a selected PolypGen subset rather than the complete 23-sequence benchmark.

[P00725 | 1002:1361 | NORMAL_TEXT | LIST id=kix.9bewq29w8lvv level=0]
A Viewpoint-Aware Framework for Polyp Segmentation — VANetCai et al., 2024, Medical Image Analysis.This is a peer-reviewed polyp-segmentation paper that includes PolypGen-related evaluation. It is useful as a strong conventional segmentation reference, although its exact split and evaluation protocol need to be checked before direct numerical comparison.

[P00726 | 1361:1671 | NORMAL_TEXT | LIST id=kix.9bewq29w8lvv level=0]
Polyp SAM 2: Advancing Zero-Shot Polyp Segmentation in Colorectal Cancer Detection2024.This paper evaluates SAM 2-based zero-shot polyp segmentation and includes PolypGen among its datasets. Check whether its PolypGen evaluation uses the whole video sequence set or extracted frames before comparing scores.

[P00727 | 1671:1704 | HEADING_2]
Original PolypGen dataset papers

[P00728 | 1704:1980 | NORMAL_TEXT | LIST id=kix.t5h5d8oi4ieq level=0]
A Multi-Centre Polyp Detection and Segmentation Dataset for Generalisability AssessmentAli et al., 2021/2023.This is the original PolypGen dataset paper. Use it as the authoritative source for dataset construction, centres, annotation, sequence data and dataset statistics.

[P00729 | 1980:2312 | NORMAL_TEXT | LIST id=kix.t5h5d8oi4ieq level=0]
Assessing Generalisability of Deep Learning-Based Polyp Detection and Segmentation Methods Through a Computer Vision ChallengeAli et al., 2024, Scientific Reports.This reports the EndoCV challenge evaluation associated with PolypGen and is useful for understanding the original generalization benchmark and participating methods.

[P00730 | 2312:2352 | HEADING_2]
Related video-polyp-segmentation papers

[P00731 | 2352:2516 | NORMAL_TEXT]
These are relevant for video segmentation and temporal modelling, but they may use SUN-SEG or other video datasets rather than the exact Kaggle PolypGen benchmark.

[P00732 | 2516:2778 | NORMAL_TEXT | LIST id=kix.ld8362pa1gm1 level=0]
VP-SAM: Taming Segment Anything Model for Video Polyp Segmentation Fang et al., ECCV 2024. Strong SAM-based video-polyp-segmentation reference. Useful for architecture and temporal-method comparison, but not automatically a directly comparable PolypGen score.

[P00733 | 2778:2972 | NORMAL_TEXT | LIST id=kix.ld8362pa1gm1 level=0]
Video Polyp Segmentation Using Implicit Networks Dahan et al., 2024. Focuses on temporal coherence in video polyp segmentation. Useful for temporal methodology and temporal evaluation design.

[P00734 | 2972:3232 | NORMAL_TEXT | LIST id=kix.ld8362pa1gm1 level=0]
Video Polyp Segmentation: A Deep Learning Perspective Ji et al., 2022. Introduces the SUN-SEG benchmark and provides a broad video-polyp-segmentation comparison. It is useful background, but SUN-SEG results should not be mixed directly with PolypGen scores.

[P00735 | 3232:3267 | HEADING_2]
Best papers to cite in your report

[P00736 | 3267:3318 | NORMAL_TEXT]
For a concise SOTA section, prioritize these five:

[P00737 | 3318:3364 | NORMAL_TEXT | LIST id=kix.35rs1mbxlj16 level=0]
Ali et al. — original PolypGen dataset paper.

[P00738 | 3364:3418 | NORMAL_TEXT | LIST id=kix.35rs1mbxlj16 level=0]
Ali et al. — EndoCV generalizability challenge paper.

[P00739 | 3418:3478 | NORMAL_TEXT | LIST id=kix.35rs1mbxlj16 level=0]
Mansoori et al. — YOLO-SAM 2, direct 23-sequence benchmark.

[P00740 | 3478:3546 | NORMAL_TEXT | LIST id=kix.35rs1mbxlj16 level=0]
Li et al. — HAT-SAM3, recent cross-dataset foundation-model result.

[P00741 | 3546:3608 | NORMAL_TEXT | LIST id=kix.35rs1mbxlj16 level=0]
Wu et al. — DepthPolyp, recent lightweight real-video result.

[P00742 | 3608:3711 | NORMAL_TEXT]
The strongest directly verified full-23-sequence score remains YOLO-SAM 2: mDice 0.808 and mIoU 0.678.

[P00743 | 3711:3712 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

## My_Paper (t.oqla2zyvac27)

[P00744 | 1:9 | NORMAL_TEXT]
Theory:

[P00745 | 9:10 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00746 | 13:21 | NORMAL_TEXT | TABLE row=0 col=0]
Problem

[P00747 | 22:47 | NORMAL_TEXT | TABLE row=0 col=1]
How my method handles it

[P00748 | 48:54 | NORMAL_TEXT | TABLE row=0 col=2]
Risks

[P00749 | 56:87 | NORMAL_TEXT | TABLE row=1 col=0]
Fast motion / viewpoint change

[P00750 | 88:133 | NORMAL_TEXT | TABLE row=1 col=1]
Temporal state keeps past object information

[P00751 | 133:165 | NORMAL_TEXT | TABLE row=1 col=1]
Optional optical-flow alignment

[P00752 | 166:199 | NORMAL_TEXT | TABLE row=1 col=2]
Misaligned features; flow errors

[P00753 | 201:216 | NORMAL_TEXT | TABLE row=2 col=0]
Blur / defocus

[P00754 | 217:275 | NORMAL_TEXT | TABLE row=2 col=1]
Uses information accumulated from previous clearer frames

[P00755 | 276:319 | NORMAL_TEXT | TABLE row=2 col=2]
Long blur periods can still corrupt memory

[P00756 | 321:356 | NORMAL_TEXT | TABLE row=3 col=0]
Reflections / illumination changes

[P00757 | 357:412 | NORMAL_TEXT | TABLE row=3 col=1]
Temporal smoothing reduces reliance on one noisy frame

[P00758 | 413:451 | NORMAL_TEXT | TABLE row=3 col=2]
Persistent artifacts remain difficult

[P00759 | 453:486 | NORMAL_TEXT | TABLE row=4 col=0]
Secretions / bubbles / occlusion

[P00760 | 487:558 | NORMAL_TEXT | TABLE row=4 col=1]
Recurrent state preserves object history through temporary obstruction

[P00761 | 559:606 | NORMAL_TEXT | TABLE row=4 col=2]
Long occlusion causes stale or forgotten state

[P00762 | 608:639 | NORMAL_TEXT | TABLE row=5 col=0]
Low contrast / weak boundaries

[P00763 | 640:664 | NORMAL_TEXT | TABLE row=5 col=1]
Previous-frame context 

[P00764 | 664:700 | NORMAL_TEXT | TABLE row=5 col=1]
=> support ambiguous current frames

[P00765 | 701:743 | NORMAL_TEXT | TABLE row=5 col=2]
No explicit boundary-refinement mechanism

[P00766 | 745:766 | NORMAL_TEXT | TABLE row=6 col=0]
Scale / shape change

[P00767 | 767:825 | NORMAL_TEXT | TABLE row=6 col=1]
State carries object identity across changing appearances

[P00768 | 826:868 | NORMAL_TEXT | TABLE row=6 col=2]
Too much smoothing causes slow adaptation

[P00769 | 870:889 | NORMAL_TEXT | TABLE row=7 col=0]
Partial visibility

[P00770 | 890:958 | NORMAL_TEXT | TABLE row=7 col=1]
Memory provides information about previously visible object regions

[P00771 | 959:990 | NORMAL_TEXT | TABLE row=7 col=2]
May hallucinate hidden regions

[P00772 | 992:1021 | NORMAL_TEXT | TABLE row=8 col=0]
Disappearance / reappearance

[P00773 | 1022:1073 | NORMAL_TEXT | TABLE row=8 col=1]
Persistent state can help recover the object later

[P00774 | 1074:1107 | NORMAL_TEXT | TABLE row=8 col=2]
Old information decays over time

[P00775 | 1109:1145 | NORMAL_TEXT | TABLE row=9 col=0]
Error propagation / snowball effect

[P00776 | 1146:1206 | NORMAL_TEXT | TABLE row=9 col=1]
Reliability gate limits updates from unreliable predictions

[P00777 | 1207:1251 | NORMAL_TEXT | TABLE row=9 col=2]
Wrong predictions can still appear reliable

[P00778 | 1253:1271 | NORMAL_TEXT | TABLE row=10 col=0]
Bad memory update

[P00779 | 1272:1338 | NORMAL_TEXT | TABLE row=10 col=1]
Reliability score controls how strongly new features enter memory

[P00780 | 1339:1390 | NORMAL_TEXT | TABLE row=10 col=2]
Reliability signals are heuristic/self-referential

[P00781 | 1392:1412 | NORMAL_TEXT | TABLE row=11 col=0]
Background collapse

[P00782 | 1413:1472 | NORMAL_TEXT | TABLE row=11 col=1]
Caps background writes; preserves foreground memory longer

[P00783 | 1473:1521 | NORMAL_TEXT | TABLE row=11 col=2]
Can retain foreground when object is truly gone

[P00784 | 1523:1553 | NORMAL_TEXT | TABLE row=12 col=0]
Stale / redundant memory bank

[P00785 | 1554:1624 | NORMAL_TEXT | TABLE row=12 col=1]
Replaces discrete past-frame bank with one continuously updated state

[P00786 | 1625:1668 | NORMAL_TEXT | TABLE row=12 col=2]
Compression may discard useful rare frames

[P00787 | 1670:1687 | NORMAL_TEXT | TABLE row=13 col=0]
Tracking failure

[P00788 | 1688:1726 | NORMAL_TEXT | TABLE row=13 col=1]
Optional YOLO re-prompt / state reset

[P00789 | 1727:1782 | NORMAL_TEXT | TABLE row=13 col=2]
Depends on detector quality; adds external supervision

[P00790 | 1784:1806 | NORMAL_TEXT | TABLE row=14 col=0]
Dense annotation cost

[P00791 | 1807:1864 | NORMAL_TEXT | TABLE row=14 col=1]
One initial prompt can propagate masks through the video

[P00792 | 1865:1908 | NORMAL_TEXT | TABLE row=14 col=2]
Still needs GT masks for proper evaluation

[P00793 | 1910:1936 | NORMAL_TEXT | TABLE row=15 col=0]
Domain / device variation

[P00794 | 1937:2002 | NORMAL_TEXT | TABLE row=15 col=1]
Training-free frozen MedSAM2 avoids dataset-specific fine-tuning

[P00795 | 2003:2047 | NORMAL_TEXT | TABLE row=15 col=2]
Method does not directly solve domain shift

[P00796 | 2048:2049 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00797 | 2049:2402 | NORMAL_TEXT]
Endoscopy videos contain transiently unreliable observations and long temporal dependencies, while SAM2-style propagation is vulnerable to stale memory and error accumulation. The method tests whether continuous recurrent temporal consolidation, followed by controlled memory updates, provides a more robust alternative to a discrete fixed memory bank.

[P00798 | 2402:2403 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00799 | 2403:2417 | NORMAL_TEXT]
Contribution:

[P00800 | 2417:2598 | NORMAL_TEXT]
A training-free recurrent memory strategy for causal medical video segmentation that continuously consolidates past observations instead of relying on a discrete fixed memory bank.

[P00801 | 2598:2643 | NORMAL_TEXT | LIST id=kix.8tvqnmwmfwdq level=0]
Recurrent temporal memory for frozen MedSAM2

[P00802 | 2643:2645 | NORMAL_TEXT]
[INLINE_OBJECT kix.23v4mgm46tzk]

[P00803 | 2645:2680 | NORMAL_TEXT | LIST id=kix.8tvqnmwmfwdq level=0]
Reliability_aware memory updating:

[P00804 | 2680:2873 | NORMAL_TEXT | LIST id=kix.c7exajidig01 level=0]
Estimate whether the current prediction is trustworthy using mask confidence, object presence, identity consistency, temporal consistency, area plausibility, and optionally detector agreement.

[P00805 | 2873:2963 | NORMAL_TEXT | LIST id=kix.c7exajidig01 level=0]
Reduce memory updates from unreliable frames to limit drift / snowball error propagation.

## Literature Review (t.755pewb2bpnb)

[P00806 | 1:104 | HEADING_1]
Path 1: MedSAM2 modification specifically for endoscopy video for adenoid and nasopharynx segmentation

[P00807 | 104:450 | NORMAL_TEXT]
Existing AI methods assess adenoid hypertrophy from selected still nasopharyngoscopy images. We introduce a video-based benchmark retaining diagnostically difficult frames and investigate whether temporal segmentation produces more robust adenoid/nasopharyngeal-airway masks and more reliable hypertrophy quantification than frame-wise analysis.

[P00808 | 450:451 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00809 | 451:624 | NORMAL_TEXT]
Research question: Can temporal information make automated adenoid hypertrophy quantification robust when individual nasopharyngoscopy frames are diagnostically unreliable?

[P00810 | 624:685 | NORMAL_TEXT]
Benefits of endoscopy videos instead of just distinct images

[P00811 | 685:912 | NORMAL_TEXT]
Challenges of endoscopy video on (adenoid/nasopharynx) segmentation and previous works: Motion, blur, reflections, illumination variation, low contrast, secretions, partial exposure, scale/viewpoint change and annotation cost.

[P00812 | 912:941 | NORMAL_TEXT | LIST id=kix.nmykc1e0yeny level=0]
motion/viewpoint instability

[P00813 | 941:1049 | NORMAL_TEXT | LIST id=kix.nmykc1e0yeny level=0]
image-quality degradation => temporal information (can also handle the dense annotation limitation problem)

[P00814 | 1049:1091 | NORMAL_TEXT | LIST id=kix.nmykc1e0yeny level=0]
weak anatomical boundaries - low-contrast

[P00815 | 1091:1124 | NORMAL_TEXT | LIST id=kix.nmykc1e0yeny level=0]
partial visibility/disappearance

[P00816 | 1124:1240 | NORMAL_TEXT | LIST id=kix.nmykc1e0yeny level=1]
can partly be mitigated by video object segmentation per object (put in the mask of the object into as mask prompt)

[P00817 | 1240:1275 | NORMAL_TEXT | LIST id=kix.nmykc1e0yeny level=0]
error propagation - mostly tackled

[P00818 | 1275:1302 | NORMAL_TEXT | LIST id=kix.nmykc1e0yeny level=1]
utilisation of memory bank

[P00819 | 1302:1347 | NORMAL_TEXT | LIST id=kix.3vso30vtv1o7 level=0]
Rapid camera motion, consecutive bad frames:

[P00820 | 1347:1349 | NORMAL_TEXT]
[INLINE_OBJECT kix.w1iyofhburaq]

[P00821 | 1349:1644 | NORMAL_TEXT | LIST id=kix.szbjjdfr3ggi level=0]
[SALI: Short-term Alignment and Long-term Interaction Network for Colonoscopy Video Polyp Segmentation](https://papers.miccai.org/miccai-2024/paper/2092_paper.pdf): “endoscope’s fast moving and close-up observing make the current methods suffer from large spatial incoherence and continuous low-quality frames, and thus yield limited segmentation accuracy”

[P00822 | 1644:1879 | NORMAL_TEXT | LIST id=kix.szbjjdfr3ggi level=1]
enhance adjacent feature consistency => “Short-term Alignment Module” - “learns spatial-aligned features of adjacent frames via deformable convolution and further harmonizes them to capture more stable short-term polyp representation”

[P00823 | 1879:2151 | NORMAL_TEXT | LIST id=kix.szbjjdfr3ggi level=1]
rebuild reliable polyp representation => “Long-term Interaction Module” - “stores the historical polyp representations as a long-term memory bank, and explores the retrospective relations to interactively rebuild more reliable polyp features for the current segmentation”

[P00824 | 2151:2218 | NORMAL_TEXT | LIST id=kix.3vso30vtv1o7 level=0]
Weak-tissue boundaries - low contrast between polyp and background

[P00825 | 2218:2511 | NORMAL_TEXT | LIST id=kix.va1hkx8jxxbz level=0]
[An Embedding-Unleashing Video Polyp Segmentation Framework via Region Linking and Scale Alignment](https://drive.google.com/drive/folders/1dbgp3FdheII29qIuu70wLke0mF8vmFwz?hl=vi): “accurate and real-time video polyp segmentation (VPS) is a very challenging task due to low contrast between background and polyps and frame-to-frame dramatic variations in colonoscopy videos”

[P00826 | 2511:2562 | NORMAL_TEXT | LIST id=kix.va1hkx8jxxbz level=0]
[AI-Enabled Mucus Segmentation in Nasal Endoscopy](https://onlinelibrary.wiley.com/doi/epdf/10.1002/lio2.70467): 

[P00827 | 2562:2583 | NORMAL_TEXT | LIST id=kix.3vso30vtv1o7 level=0]
Annotation scarcity:

[P00828 | 2583:2731 | NORMAL_TEXT | LIST id=kix.u9qjkjucdwui level=0]
[Video Polyp Segmentation: A Deep Learning Perspective / SUN-SEG](https://arxiv.org/pdf/2203.14291): the need to exploit TEMPORAL information rather than treating frames independently

[P00829 | 2731:2864 | NORMAL_TEXT | LIST id=kix.u9qjkjucdwui level=0]
[Deep Learning-Based Quantification of Adenoid Hypertrophy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11687100/): “scarcity of large, high-quality datasets in specialized medical fields”	

[P00830 | 2864:2985 | NORMAL_TEXT | LIST id=kix.u9qjkjucdwui level=1]
To address these issues, techniques such as transfer learning and ensemble learning have emerged as effective solutions.

[P00831 | 2985:3158 | NORMAL_TEXT | LIST id=kix.u9qjkjucdwui level=2]
Transfer learning allows a model trained on one task to be adapted for a related task with a small dataset, leveraging knowledge gained from large-scale pre-trained models.

[P00832 | 3158:3333 | NORMAL_TEXT | LIST id=kix.u9qjkjucdwui level=2]
Ensemble learning, which combines multiple models to improve overall performance, has been used to aggregate predictions, thereby increasing accuracy and reduced overfitting.

[P00833 | 3333:3364 | NORMAL_TEXT]
Dataset (need to be explicit):

[P00834 | 3364:3382 | NORMAL_TEXT]
reference source:

[P00835 | 3382:3390 | NORMAL_TEXT | LIST id=kix.6bov778uhnd7 level=0]
[SUN-SEG](https://arxiv.org/pdf/2203.14291)

[P00836 | 3390:3399 | NORMAL_TEXT | LIST id=kix.6bov778uhnd7 level=0]
[MIB-ANet](https://europepmc.org/api/getPdf?pmcid=PMC10140414)

[P00837 | 3399:3457 | NORMAL_TEXT | LIST id=kix.6bov778uhnd7 level=0]
[Deep Learning-Based Quantification of Adenoid Hypertrophy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11687100/)

[P00838 | 3457:3458 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00839 | 3461:3467 | NORMAL_TEXT | TABLE row=0 col=0]
Level

[P00840 | 3468:3482 | NORMAL_TEXT | TABLE row=0 col=1]
What to store

[P00841 | 3483:3487 | NORMAL_TEXT | TABLE row=0 col=2]
Why

[P00842 | 3489:3511 | NORMAL_TEXT | TABLE row=1 col=0]
Patient / examination

[P00843 | 3512:3620 | NORMAL_TEXT | TABLE row=1 col=1]
pseudonymous patient ID, video ID, age range, clinical AH grade, device/site, optional AHI/PSG if available

[P00844 | 3621:3673 | NORMAL_TEXT | TABLE row=1 col=2]
clinical endpoint + correct patient-level splitting

[P00845 | 3675:3688 | NORMAL_TEXT | TABLE row=2 col=0]
Frame / clip

[P00846 | 3689:3797 | NORMAL_TEXT | TABLE row=2 col=1]
timestamp, diagnostic-view flag, visibility, blur, secretion, specular reflection, motion, partial exposure

[P00847 | 3798:3841 | NORMAL_TEXT | TABLE row=2 col=2]
directly study the difficult-frame problem

[P00848 | 3843:3856 | NORMAL_TEXT | TABLE row=3 col=0]
Segmentation

[P00849 | 3857:3921 | NORMAL_TEXT | TABLE row=3 col=1]
adenoid, nasopharynx_airway, optionally ignore/uncertain region

[P00850 | 3922:3956 | NORMAL_TEXT | TABLE row=3 col=2]
trains/evaluates the actual model

[P00851 | 3957:3974 | NORMAL_TEXT]
Process (maybe):

[P00852 | 3974:3984 | NORMAL_TEXT]
Raw video

[P00853 | 3984:3989 | NORMAL_TEXT]
   ↓

[P00854 | 3989:4170 | NORMAL_TEXT]
CVAT (video annotation, object tracking, automatic annotation using models, frame attributes, dedicated attribute-annotation mode, SAM2 video tracker => propagate through sequence)

[P00855 | 4170:4175 | NORMAL_TEXT]
   ↓

[P00856 | 4175:4215 | NORMAL_TEXT]
ENT/student marks adenoid + airway once

[P00857 | 4215:4220 | NORMAL_TEXT]
   ↓

[P00858 | 4220:4250 | NORMAL_TEXT]
SAM2 tracker propagates masks

[P00859 | 4250:4255 | NORMAL_TEXT]
   ↓

[P00860 | 4255:4274 | NORMAL_TEXT]
human fixes errors

[P00861 | 4274:4279 | NORMAL_TEXT]
   ↓

[P00862 | 4279:4296 | NORMAL_TEXT]
automatic script

[P00863 | 4296:4310 | NORMAL_TEXT]
 ├─ timestamp

[P00864 | 4310:4324 | NORMAL_TEXT]
 ├─ mask area

[P00865 | 4324:4337 | NORMAL_TEXT]
 ├─ centroid

[P00866 | 4337:4357 | NORMAL_TEXT]
 ├─ centroid motion

[P00867 | 4357:4374 | NORMAL_TEXT]
 ├─ optical flow

[P00868 | 4374:4394 | NORMAL_TEXT]
 ├─ scale variation

[P00869 | 4394:4413 | NORMAL_TEXT]
 ├─ border contact

[P00870 | 4413:4428 | NORMAL_TEXT]
 ├─ blur score

[P00871 | 4428:4447 | NORMAL_TEXT]
 └─ specular ratio

[P00872 | 4447:4452 | NORMAL_TEXT]
   ↓

[P00873 | 4452:4470 | NORMAL_TEXT]
human labels only

[P00874 | 4470:4490 | NORMAL_TEXT]
 ├─ diagnostic view

[P00875 | 4490:4504 | NORMAL_TEXT]
 ├─ secretion

[P00876 | 4504:4533 | NORMAL_TEXT]
 └─ visibility / uncertainty

[P00877 | 4533:4534 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00878 | 4534:4536 | NORMAL_TEXT]
[INLINE_OBJECT kix.yz316jbabso4]

[P00879 | 4536:4548 | NORMAL_TEXT]
Evaluation:

[P00880 | 4548:4550 | NORMAL_TEXT]
[INLINE_OBJECT kix.gl7df5yju49d]

[P00881 | 4550:4610 | NORMAL_TEXT]
[Deep Learning-Based Quantification of Adenoid Hypertrophy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11687100/): 

[P00882 | 4610:4911 | NORMAL_TEXT | LIST id=kix.ianu7jfs69wl level=0]
“nasopharyngoscopy images were manually annotated using MATLAB’s imfreehand tool. Two experienced otolaryngologists delineated the boundaries of the adenoid and the nasopharyngeal space to create region-specific masks. These annotations were then converted into binary masks for subsequent analysis.”

[P00883 | 4911:5160 | NORMAL_TEXT | LIST id=kix.ianu7jfs69wl level=0]
“The final analysis relied on the overlapping region of their annotations. If the difference between the two otolaryngologists exceeded 5%, a senior otolaryngologist with over 30 years of clinical experience was consulted to perform re-annotation.”

[P00884 | 5160:5342 | NORMAL_TEXT | LIST id=kix.ianu7jfs69wl level=0]
Using these annotations, the algorithm automatically computed the A/N ratio and categorized AH into three levels based on this ratio: small (<50%), medium (50–75%), and large (>75%)

[P00885 | 5342:5350 | NORMAL_TEXT]
Method:

[P00886 | 5350:5351 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00887 | 5354:5364 | NORMAL_TEXT | TABLE row=0 col=0]
Component

[P00888 | 5365:5384 | NORMAL_TEXT | TABLE row=0 col=1]
Minimum for MICCAI

[P00889 | 5385:5402 | NORMAL_TEXT | TABLE row=0 col=2]
Stronger version

[P00890 | 5404:5422 | NORMAL_TEXT | TABLE row=1 col=0]
Clinical question

[P00891 | 5423:5479 | NORMAL_TEXT | TABLE row=1 col=1]
Clearly define why still-image analysis is insufficient

[P00892 | 5480:5571 | NORMAL_TEXT | TABLE row=1 col=2]
Show that difficult/non-diagnostic frames occur frequently and affect clinical measurement

[P00893 | 5573:5581 | NORMAL_TEXT | TABLE row=2 col=0]
Dataset

[P00894 | 5582:5621 | NORMAL_TEXT | TABLE row=2 col=1]
Patient-level nasopharyngoscopy videos

[P00895 | 5622:5658 | NORMAL_TEXT | TABLE row=2 col=2]
Multi-device or multi-center videos

[P00896 | 5660:5669 | NORMAL_TEXT | TABLE row=3 col=0]
Subjects

[P00897 | 5670:5746 | NORMAL_TEXT | TABLE row=3 col=1]
Preferably ~80–150+ patients rather than huge frame count from few patients

[P00898 | 5747:5803 | NORMAL_TEXT | TABLE row=3 col=2]
150–300+ patients with meaningful severity distribution

[P00899 | 5805:5813 | NORMAL_TEXT | TABLE row=4 col=0]
Targets

[P00900 | 5814:5843 | NORMAL_TEXT | TABLE row=4 col=1]
adenoid + nasopharynx_airway

[P00901 | 5844:5887 | NORMAL_TEXT | TABLE row=4 col=2]
+ diagnostic-view / visibility annotations

[P00902 | 5889:5907 | NORMAL_TEXT | TABLE row=5 col=0]
Video annotations

[P00903 | 5908:5949 | NORMAL_TEXT | TABLE row=5 col=1]
Dense clips from a representative subset

[P00904 | 5950:6010 | NORMAL_TEXT | TABLE row=5 col=2]
Dense difficult-event clips + sparse coverage of all videos

[P00905 | 6012:6034 | NORMAL_TEXT | TABLE row=6 col=0]
Difficulty attributes

[P00906 | 6035:6105 | NORMAL_TEXT | TABLE row=6 col=1]
motion, blur, secretion, reflection, incomplete exposure / visibility

[P00907 | 6106:6171 | NORMAL_TEXT | TABLE row=6 col=2]
Objective automatic scores + human labels + inter-rater analysis

[P00908 | 6173:6186 | NORMAL_TEXT | TABLE row=7 col=0]
Ground truth

[P00909 | 6187:6212 | NORMAL_TEXT | TABLE row=7 col=1]
Clinician-reviewed masks

[P00910 | 6213:6281 | NORMAL_TEXT | TABLE row=7 col=2]
Two independent ENT annotators + senior adjudication on test subset

[P00911 | 6283:6295 | NORMAL_TEXT | TABLE row=8 col=0]
Clinical GT

[P00912 | 6296:6337 | NORMAL_TEXT | TABLE row=8 col=1]
hypertrophy grade / obstruction estimate

[P00913 | 6338:6399 | NORMAL_TEXT | TABLE row=8 col=2]
Independent reader panel with junior/intermediate/senior ENT

[P00914 | 6401:6411 | NORMAL_TEXT | TABLE row=9 col=0]
Splitting

[P00915 | 6412:6441 | NORMAL_TEXT | TABLE row=9 col=1]
patient-level train/val/test

[P00916 | 6442:6477 | NORMAL_TEXT | TABLE row=9 col=2]
External-site/device held-out test

[P00917 | 6479:6494 | NORMAL_TEXT | TABLE row=10 col=0]
Frame baseline

[P00918 | 6495:6524 | NORMAL_TEXT | TABLE row=10 col=1]
Strong 2D segmentation model

[P00919 | 6525:6587 | NORMAL_TEXT | TABLE row=10 col=2]
nnU-Net / modern medical segmentation + SAM/MedSAM image mode

[P00920 | 6589:6604 | NORMAL_TEXT | TABLE row=11 col=0]
Video baseline

[P00921 | 6605:6620 | NORMAL_TEXT | TABLE row=11 col=1]
MedSAM2 / SAM2

[P00922 | 6621:6651 | NORMAL_TEXT | TABLE row=11 col=2]
+ another temporal VOS method

[P00923 | 6653:6669 | NORMAL_TEXT | TABLE row=12 col=0]
Core experiment

[P00924 | 6670:6700 | NORMAL_TEXT | TABLE row=12 col=1]
frame model vs temporal model

[P00925 | 6701:6761 | NORMAL_TEXT | TABLE row=12 col=2]
control backbone/prompt so temporal information is isolated

[P00926 | 6763:6784 | NORMAL_TEXT | TABLE row=13 col=0]
Segmentation metrics

[P00927 | 6785:6796 | NORMAL_TEXT | TABLE row=13 col=1]
Dice + IoU

[P00928 | 6797:6828 | NORMAL_TEXT | TABLE row=13 col=2]
+ HD95/ASSD or boundary metric

[P00929 | 6830:6847 | NORMAL_TEXT | TABLE row=14 col=0]
Temporal metrics

[P00930 | 6848:6878 | NORMAL_TEXT | TABLE row=14 col=1]
Temporal IoU / area variation

[P00931 | 6879:6941 | NORMAL_TEXT | TABLE row=14 col=2]
failure duration + recovery after blur/occlusion/reappearance

[P00932 | 6943:6960 | NORMAL_TEXT | TABLE row=15 col=0]
Clinical metrics

[P00933 | 6961:7004 | NORMAL_TEXT | TABLE row=15 col=1]
A/N or obstruction error + grade agreement

[P00934 | 7005:7058 | NORMAL_TEXT | TABLE row=15 col=2]
MAE, ICC, weighted κ, Bland–Altman where appropriate

[P00935 | 7060:7081 | NORMAL_TEXT | TABLE row=16 col=0]
Attribute evaluation

[P00936 | 7082:7110 | NORMAL_TEXT | TABLE row=16 col=1]
overall vs difficult frames

[P00937 | 7111:7168 | NORMAL_TEXT | TABLE row=16 col=2]
statistical comparison across motion/blur/secretion/etc.

[P00938 | 7170:7181 | NORMAL_TEXT | TABLE row=17 col=0]
Statistics

[P00939 | 7182:7199 | NORMAL_TEXT | TABLE row=17 col=1]
mean + variation

[P00940 | 7200:7257 | NORMAL_TEXT | TABLE row=17 col=2]
95% CI + paired statistical tests at patient/video level

[P00941 | 7259:7275 | NORMAL_TEXT | TABLE row=18 col=0]
Reproducibility

[P00942 | 7276:7317 | NORMAL_TEXT | TABLE row=18 col=1]
detailed acquisition/annotation/protocol

[P00943 | 7318:7383 | NORMAL_TEXT | TABLE row=18 col=2]
release dataset or at least test subset/code/annotation protocol

[P00944 | 7384:7385 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00945 | 7385:7432 | NORMAL_TEXT]
Experiment: frame-only vs temporal/video model

[P00946 | 7432:7455 | NORMAL_TEXT]
3 questions to verify:

[P00947 | 7455:7515 | NORMAL_TEXT | LIST id=kix.lgfocjtko1iz level=0]
Q1:Does temporal information improve segmentation overall? 

[P00948 | 7515:7563 | NORMAL_TEXT | LIST id=kix.lgfocjtko1iz level=0]
Q2:Does it help especially on difficult frames?

[P00949 | 7563:7653 | NORMAL_TEXT | LIST id=kix.lgfocjtko1iz level=0]
Q3​:Does that segmentation improvement actually improve clinical hypertrophy measurement?

[P00950 | 7653:7714 | HEADING_1]
Path 2: Stop modifying the bank, modify the MEMORY attention

[P00951 | 7714:7817 | NORMAL_TEXT]
Research question: Given the memory we already have, how should the current frame retrieve and use it?

[P00952 | 7817:7832 | NORMAL_TEXT]
Previous work:

[P00953 | 7832:7943 | NORMAL_TEXT]
[SAM2Long](https://arxiv.org/pdf/2410.16268): memory-attention modulation, assigning different importance to memory entries during cross-attention

[P00954 | 7943:8071 | NORMAL_TEXT]
[EdgeTAM (CVPR 2025)](https://arxiv.org/pdf/2501.07256): redesign memory representation/attention for efficiency (on-device track, not closely related to my field)

[P00955 | 8071:8175 | NORMAL_TEXT]
[Efficient-SAM2](https://arxiv.org/pdf/2602.08224): only a subset of memory tokens contribute strongly and proposes sparse memory retrieval

[P00956 | 8175:8248 | NORMAL_TEXT]
[MPG-SAM2, ICCV 2025](https://arxiv.org/pdf/2501.13667): sophisticated historical/global attention mechanism

[P00957 | 8248:8433 | NORMAL_TEXT]
[Unlocking the Power of SAM 2 for Few-Shot Segmentation](https://arxiv.org/pdf/2505.14100): SAM2-based few-shot work has already proposed Support-Calibrated Memory Attention to suppress harmful background memory features

[P00958 | 8433:8434 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P00959 | 8437:8443 | NORMAL_TEXT | TABLE row=0 col=0]
Paper

[P00960 | 8444:8452 | NORMAL_TEXT | TABLE row=0 col=1]
Problem

[P00961 | 8453:8485 | NORMAL_TEXT | TABLE row=0 col=2]
What part of attention changes?

[P00962 | 8486:8496 | NORMAL_TEXT | TABLE row=0 col=3]
Mechanism

[P00963 | 8497:8523 | NORMAL_TEXT | TABLE row=0 col=4]
Limitation relevant to us

[P00964 | 8525:8534 | NORMAL_TEXT | TABLE row=1 col=0]
SAM2Long

[P00965 | 8535:8555 | NORMAL_TEXT | TABLE row=1 col=1]
unreliable memories

[P00966 | 8556:8577 | NORMAL_TEXT | TABLE row=1 col=2]
memory-key weighting

[P00967 | 8578:8607 | NORMAL_TEXT | TABLE row=1 col=3]
reliability-based modulation

[P00968 | 8608:8628 | NORMAL_TEXT | TABLE row=1 col=4]
heuristic weighting

[P00969 | 8630:8638 | NORMAL_TEXT | TABLE row=2 col=0]
EdgeTAM

[P00970 | 8639:8651 | NORMAL_TEXT | TABLE row=2 col=1]
computation

[P00971 | 8652:8677 | NORMAL_TEXT | TABLE row=2 col=2]
representation/attention

[P00972 | 8678:8706 | NORMAL_TEXT | TABLE row=2 col=3]
efficient memory processing

[P00973 | 8707:8741 | NORMAL_TEXT | TABLE row=2 col=4]
efficiency rather than robustness

[P00974 | 8743:8758 | NORMAL_TEXT | TABLE row=3 col=0]
Efficient-SAM2

[P00975 | 8759:8776 | NORMAL_TEXT | TABLE row=3 col=1]
redundant tokens

[P00976 | 8777:8787 | NORMAL_TEXT | TABLE row=3 col=2]
retrieval

[P00977 | 8788:8811 | NORMAL_TEXT | TABLE row=3 col=3]
sparse token selection

[P00978 | 8812:8859 | NORMAL_TEXT | TABLE row=3 col=4]
doesn't necessarily solve wrong correspondence

[P00979 | 8861:8870 | NORMAL_TEXT | TABLE row=4 col=0]
MPG-SAM2

[P00980 | 8871:8892 | NORMAL_TEXT | TABLE row=4 col=1]
insufficient context

[P00981 | 8893:8911 | NORMAL_TEXT | TABLE row=4 col=2]
attention/context

[P00982 | 8912:8944 | NORMAL_TEXT | TABLE row=4 col=3]
historical + global information

[P00983 | 8945:8979 | NORMAL_TEXT | TABLE row=4 col=4]
larger mechanism / different task

[P00984 | 8981:9017 | NORMAL_TEXT | TABLE row=5 col=0]
Support-Calibrated Memory Attention

[P00985 | 9018:9045 | NORMAL_TEXT | TABLE row=5 col=1]
harmful background support

[P00986 | 9046:9068 | NORMAL_TEXT | TABLE row=5 col=2]
attention calibration

[P00987 | 9069:9106 | NORMAL_TEXT | TABLE row=5 col=3]
suppress irrelevant support features

[P00988 | 9107:9124 | NORMAL_TEXT | TABLE row=5 col=4]
few-shot setting

[P00989 | 9125:9199 | NORMAL_TEXT]
TODO: first discover a specific failure mechanism inside memory attention

[P00990 | 9199:9266 | HEADING_1]
Path 3: Do we actually need a feature memory bank? (Experimenting)

[P00991 | 9266:9417 | NORMAL_TEXT]
Research question: can a compact recurrent spatial memory state replace SAM2/MedSAM2’s multi-frame memory bank while retaining useful VOS performance?

[P00992 | 9417:9433 | NORMAL_TEXT]
Previous Works:

[P00993 | 9433:9453 | NORMAL_TEXT | LIST id=kix.i7h4y1tgy1wg level=0]
LiVOS (CVPR 2025): 

[P00994 | 9453:9528 | NORMAL_TEXT | LIST id=kix.f7l73s22jw8t level=0]
algebraically reformulate matching into a recurrent constant-size 2D state

[P00995 | 9528:9590 | NORMAL_TEXT]
S_t = f(S_(t-1), F_t) rather than M_t = {F_(t-1), F_(t-2), …}

[P00996 | 9590:9664 | NORMAL_TEXT | LIST id=kix.qalcl2imf2ds level=0]
do NOT necessarily need to explicitly retain individual historical frames

[P00997 | 9664:9715 | NORMAL_TEXT | LIST id=kix.qalcl2imf2ds level=0]
LEARNS how to construct and gate the current state

[P00998 | 9715:9734 | NORMAL_TEXT | LIST id=kix.i7h4y1tgy1wg level=0]
Cutie (CVPR 2024):

[P00999 | 9734:9802 | NORMAL_TEXT | LIST id=kix.b2czlcstnbe2 level=0]
maintains a COMPACT object-level representation - streaming average

[P01000 | 9802:9832 | NORMAL_TEXT]
S_t = Aggregate(S_(t-1), F_t)

[P01001 | 9832:9922 | NORMAL_TEXT | LIST id=kix.k7h199uvg7re level=0]
accumulated/averaged (smoothed) representations can preserve useful object representation

[P01002 | 9922:9940 | NORMAL_TEXT | LIST id=kix.i7h4y1tgy1wg level=0]
RMem (CVPR 2024):

[P01003 | 9940:9979 | NORMAL_TEXT | LIST id=kix.nsccq8xqzjg level=0]
more historical frames = better memory

[P01004 | 9979:9999 | NORMAL_TEXT | LIST id=kix.nsccq8xqzjg level=0]
+LiVOS => intuition

[P01005 | 9999:10018 | NORMAL_TEXT | LIST id=kix.i7h4y1tgy1wg level=0]
SALI, MICCAI 2024:

[P01006 | 10018:10085 | NORMAL_TEXT | LIST id=kix.d6ztqdv6ey37 level=0]
short-term feature alignment + long-term historical representation

[P01007 | 10085:10086 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

## Experiment Results (t.3o7ufi7oq5wt)

[P01008 | 1:13 | HEADING_2]
Experiments

[P01009 | 13:15 | NORMAL_TEXT]
[INLINE_OBJECT kix.44ckh8lwzh3m]

[P01010 | 15:26 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=0]
Frame-only

[P01011 | 26:48 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=1]
manual box generation

[P01012 | 48:69 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=1]
YOLv8 box generation

[P01013 | 69:85 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=0]
default MedSAM2

[P01014 | 85:135 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=1]
add first positive box prompt + naive memory bank

[P01015 | 135:208 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=1]
anchor mask + naive memory bank (object-pointer from most recent frames)

[P01016 | 208:290 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=1]
anchor mask + IoU-confidence memory bank (object-pointer from most recent frames)

[P01017 | 290:306 | HEADING_2]
Overall Results

[P01018 | 309:316 | NORMAL_TEXT | TABLE row=0 col=0]
Method

[P01019 | 317:328 | NORMAL_TEXT | TABLE row=0 col=1]
Frame Dice

[P01020 | 329:339 | NORMAL_TEXT | TABLE row=0 col=2]
Frame IoU

[P01021 | 340:354 | NORMAL_TEXT | TABLE row=0 col=3]
Sequence Dice

[P01022 | 355:368 | NORMAL_TEXT | TABLE row=0 col=4]
Sequence IoU

[P01023 | 370:413 | NORMAL_TEXT | TABLE row=1 col=0]
Independent frames — dataset box per frame

[P01024 | 414:421 | NORMAL_TEXT | TABLE row=1 col=1]
0.9096

[P01025 | 422:429 | NORMAL_TEXT | TABLE row=1 col=2]
0.8658

[P01026 | 430:437 | NORMAL_TEXT | TABLE row=1 col=3]
0.9232

[P01027 | 438:445 | NORMAL_TEXT | TABLE row=1 col=4]
0.8834

[P01028 | 447:489 | NORMAL_TEXT | TABLE row=2 col=0]
Independent frames — YOLOv8 box per frame

[P01029 | 490:497 | NORMAL_TEXT | TABLE row=2 col=1]
0.8300

[P01030 | 498:505 | NORMAL_TEXT | TABLE row=2 col=2]
0.7915

[P01031 | 506:513 | NORMAL_TEXT | TABLE row=2 col=3]
0.7700

[P01032 | 514:521 | NORMAL_TEXT | TABLE row=2 col=4]
0.7392

[P01033 | 523:574 | NORMAL_TEXT | TABLE row=3 col=0]
VOS — first available dataset box, standard memory

[P01034 | 575:582 | NORMAL_TEXT | TABLE row=3 col=1]
0.5551

[P01035 | 583:590 | NORMAL_TEXT | TABLE row=3 col=2]
0.5178

[P01036 | 591:598 | NORMAL_TEXT | TABLE row=3 col=3]
0.6955

[P01037 | 599:606 | NORMAL_TEXT | TABLE row=3 col=4]
0.6535

[P01038 | 608:659 | NORMAL_TEXT | TABLE row=4 col=0]
VOS-per-object — anchor mask + FIFO spatial memory

[P01039 | 660:667 | NORMAL_TEXT | TABLE row=4 col=1]
0.5640

[P01040 | 668:675 | NORMAL_TEXT | TABLE row=4 col=2]
0.5276

[P01041 | 676:683 | NORMAL_TEXT | TABLE row=4 col=3]
0.7001

[P01042 | 684:691 | NORMAL_TEXT | TABLE row=4 col=4]
0.6589

[P01043 | 693:760 | NORMAL_TEXT | TABLE row=5 col=0]
VOS-per-object — anchor mask + predicted-IoU-ranked spatial memory

[P01044 | 761:768 | NORMAL_TEXT | TABLE row=5 col=1]
0.5635

[P01045 | 769:776 | NORMAL_TEXT | TABLE row=5 col=2]
0.5139

[P01046 | 777:784 | NORMAL_TEXT | TABLE row=5 col=3]
0.6864

[P01047 | 785:792 | NORMAL_TEXT | TABLE row=5 col=4]
0.6399

[P01048 | 793:809 | NORMAL_TEXT]
Interpretation:

[P01049 | 809:879 | NORMAL_TEXT | LIST id=kix.x8ea39te4ayf level=0]
The reported Dice/IoU differences are real differences in predictions

[P01050 | 879:919 | NORMAL_TEXT | LIST id=kix.26lszn25qf0 level=0]
Fresh localization helps substantially.

[P01051 | 919:920 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01052 | 923:934 | NORMAL_TEXT | TABLE row=0 col=0]
Frame Dice

[P01053 | 935:947 | NORMAL_TEXT | TABLE row=0 col=1]
Initial box

[P01054 | 948:960 | NORMAL_TEXT | TABLE row=0 col=2]
Anchor mask

[P01055 | 961:972 | NORMAL_TEXT | TABLE row=0 col=3]
Difference

[P01056 | 974:985 | NORMAL_TEXT | TABLE row=1 col=0]
All frames

[P01057 | 986:993 | NORMAL_TEXT | TABLE row=1 col=1]
0.5551

[P01058 | 994:1001 | NORMAL_TEXT | TABLE row=1 col=2]
0.5640

[P01059 | 1002:1010 | NORMAL_TEXT | TABLE row=1 col=3]
+0.0089

[P01060 | 1012:1044 | NORMAL_TEXT | TABLE row=2 col=0]
Excluding initialization frames

[P01061 | 1045:1052 | NORMAL_TEXT | TABLE row=2 col=1]
0.5515

[P01062 | 1053:1060 | NORMAL_TEXT | TABLE row=2 col=2]
0.5599

[P01063 | 1061:1069 | NORMAL_TEXT | TABLE row=2 col=3]
+0.0084

[P01064 | 1070:1071 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01065 | 1071:1072 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01066 | 1075:1086 | NORMAL_TEXT | TABLE row=0 col=0]
Hypothesis

[P01067 | 1087:1096 | NORMAL_TEXT | TABLE row=0 col=1]
Evidence

[P01068 | 1097:1108 | NORMAL_TEXT | TABLE row=0 col=2]
Assessment

[P01069 | 1110:1177 | NORMAL_TEXT | TABLE row=1 col=0]
Anchor mask + FIFO memory is better than initial box + FIFO memory

[P01070 | 1178:1238 | NORMAL_TEXT | TABLE row=1 col=1]
Frame Dice: 0.5551 → 0.5640; Sequence Dice: 0.6955 → 0.7001

[P01071 | 1239:1266 | NORMAL_TEXT | TABLE row=1 col=2]
Small average improvement.

[P01072 | 1268:1323 | NORMAL_TEXT | TABLE row=2 col=0]
Predicted-IoU-ranked memory is better than FIFO memory

[P01073 | 1324:1384 | NORMAL_TEXT | TABLE row=2 col=1]
Frame Dice: 0.5640 → 0.5635; Sequence Dice: 0.7001 → 0.6864

[P01074 | 1385:1408 | NORMAL_TEXT | TABLE row=2 col=2]
Not supported overall.

[P01075 | 1410:1467 | NORMAL_TEXT | TABLE row=3 col=0]
Frame-wise inference is better than temporal propagation

[P01076 | 1468:1547 | NORMAL_TEXT | TABLE row=3 col=1]
Dataset-box frame-wise method wins in 20 sequences, loses in 1, and ties in 2.

[P01077 | 1548:1620 | NORMAL_TEXT | TABLE row=3 col=2]
True for these experiments, but it receives a fresh box on every frame.

[P01078 | 1621:1622 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01079 | 1622:1623 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01080 | 1626:1630 | NORMAL_TEXT | TABLE row=0 col=0]
Seq

[P01081 | 1631:1637 | NORMAL_TEXT | TABLE row=0 col=1]
Frame

[P01082 | 1638:1652 | NORMAL_TEXT | TABLE row=0 col=2]
First-box VOS

[P01083 | 1653:1665 | NORMAL_TEXT | TABLE row=0 col=3]
Anchor FIFO

[P01084 | 1666:1687 | NORMAL_TEXT | TABLE row=0 col=4]
Anchor predicted-IoU

[P01085 | 1688:1711 | NORMAL_TEXT | TABLE row=0 col=5]
Frame-wise dataset box

[P01086 | 1712:1732 | NORMAL_TEXT | TABLE row=0 col=6]
Frame-wise YOLO box

[P01087 | 1734:1736 | NORMAL_TEXT | TABLE row=1 col=0]
1

[P01088 | 1737:1740 | NORMAL_TEXT | TABLE row=1 col=1]
36

[P01089 | 1741:1748 | NORMAL_TEXT | TABLE row=1 col=2]
1.0000

[P01090 | 1749:1756 | NORMAL_TEXT | TABLE row=1 col=3]
1.0000

[P01091 | 1757:1764 | NORMAL_TEXT | TABLE row=1 col=4]
1.0000

[P01092 | 1765:1772 | NORMAL_TEXT | TABLE row=1 col=5]
1.0000

[P01093 | 1773:1780 | NORMAL_TEXT | TABLE row=1 col=6]
1.0000

[P01094 | 1782:1784 | NORMAL_TEXT | TABLE row=2 col=0]
2

[P01095 | 1785:1788 | NORMAL_TEXT | TABLE row=2 col=1]
63

[P01096 | 1789:1796 | NORMAL_TEXT | TABLE row=2 col=2]
0.9014

[P01097 | 1797:1804 | NORMAL_TEXT | TABLE row=2 col=3]
0.9017

[P01098 | 1805:1812 | NORMAL_TEXT | TABLE row=2 col=4]
0.9023

[P01099 | 1813:1820 | NORMAL_TEXT | TABLE row=2 col=5]
0.9609

[P01100 | 1821:1828 | NORMAL_TEXT | TABLE row=2 col=6]
0.9146

[P01101 | 1830:1832 | NORMAL_TEXT | TABLE row=3 col=0]
3

[P01102 | 1833:1836 | NORMAL_TEXT | TABLE row=3 col=1]
15

[P01103 | 1837:1844 | NORMAL_TEXT | TABLE row=3 col=2]
0.8381

[P01104 | 1845:1852 | NORMAL_TEXT | TABLE row=3 col=3]
0.8483

[P01105 | 1853:1860 | NORMAL_TEXT | TABLE row=3 col=4]
0.8594

[P01106 | 1861:1868 | NORMAL_TEXT | TABLE row=3 col=5]
0.9424

[P01107 | 1869:1876 | NORMAL_TEXT | TABLE row=3 col=6]
0.0614

[P01108 | 1878:1880 | NORMAL_TEXT | TABLE row=4 col=0]
4

[P01109 | 1881:1884 | NORMAL_TEXT | TABLE row=4 col=1]
48

[P01110 | 1885:1892 | NORMAL_TEXT | TABLE row=4 col=2]
0.7886

[P01111 | 1893:1900 | NORMAL_TEXT | TABLE row=4 col=3]
0.7909

[P01112 | 1901:1908 | NORMAL_TEXT | TABLE row=4 col=4]
0.7913

[P01113 | 1909:1916 | NORMAL_TEXT | TABLE row=4 col=5]
0.9511

[P01114 | 1917:1924 | NORMAL_TEXT | TABLE row=4 col=6]
0.1673

[P01115 | 1926:1928 | NORMAL_TEXT | TABLE row=5 col=0]
5

[P01116 | 1929:1933 | NORMAL_TEXT | TABLE row=5 col=1]
250

[P01117 | 1934:1941 | NORMAL_TEXT | TABLE row=5 col=2]
0.5867

[P01118 | 1942:1949 | NORMAL_TEXT | TABLE row=5 col=3]
0.5828

[P01119 | 1950:1957 | NORMAL_TEXT | TABLE row=5 col=4]
0.5980

[P01120 | 1958:1965 | NORMAL_TEXT | TABLE row=5 col=5]
0.9420

[P01121 | 1966:1973 | NORMAL_TEXT | TABLE row=5 col=6]
0.9302

[P01122 | 1975:1977 | NORMAL_TEXT | TABLE row=6 col=0]
6

[P01123 | 1978:1981 | NORMAL_TEXT | TABLE row=6 col=1]
91

[P01124 | 1982:1989 | NORMAL_TEXT | TABLE row=6 col=2]
0.8443

[P01125 | 1990:1997 | NORMAL_TEXT | TABLE row=6 col=3]
0.8512

[P01126 | 1998:2005 | NORMAL_TEXT | TABLE row=6 col=4]
0.4499

[P01127 | 2006:2013 | NORMAL_TEXT | TABLE row=6 col=5]
0.9606

[P01128 | 2014:2021 | NORMAL_TEXT | TABLE row=6 col=6]
0.9451

[P01129 | 2023:2025 | NORMAL_TEXT | TABLE row=7 col=0]
7

[P01130 | 2026:2029 | NORMAL_TEXT | TABLE row=7 col=1]
48

[P01131 | 2030:2037 | NORMAL_TEXT | TABLE row=7 col=2]
1.0000

[P01132 | 2038:2045 | NORMAL_TEXT | TABLE row=7 col=3]
1.0000

[P01133 | 2046:2053 | NORMAL_TEXT | TABLE row=7 col=4]
1.0000

[P01134 | 2054:2061 | NORMAL_TEXT | TABLE row=7 col=5]
1.0000

[P01135 | 2062:2069 | NORMAL_TEXT | TABLE row=7 col=6]
1.0000

[P01136 | 2071:2073 | NORMAL_TEXT | TABLE row=8 col=0]
8

[P01137 | 2074:2077 | NORMAL_TEXT | TABLE row=8 col=1]
73

[P01138 | 2078:2085 | NORMAL_TEXT | TABLE row=8 col=2]
0.6557

[P01139 | 2086:2093 | NORMAL_TEXT | TABLE row=8 col=3]
0.6685

[P01140 | 2094:2101 | NORMAL_TEXT | TABLE row=8 col=4]
0.6399

[P01141 | 2102:2109 | NORMAL_TEXT | TABLE row=8 col=5]
0.9181

[P01142 | 2110:2117 | NORMAL_TEXT | TABLE row=8 col=6]
0.8875

[P01143 | 2119:2121 | NORMAL_TEXT | TABLE row=9 col=0]
9

[P01144 | 2122:2125 | NORMAL_TEXT | TABLE row=9 col=1]
51

[P01145 | 2126:2133 | NORMAL_TEXT | TABLE row=9 col=2]
0.7356

[P01146 | 2134:2141 | NORMAL_TEXT | TABLE row=9 col=3]
0.7258

[P01147 | 2142:2149 | NORMAL_TEXT | TABLE row=9 col=4]
0.7295

[P01148 | 2150:2157 | NORMAL_TEXT | TABLE row=9 col=5]
0.9419

[P01149 | 2158:2165 | NORMAL_TEXT | TABLE row=9 col=6]
0.3401

[P01150 | 2167:2170 | NORMAL_TEXT | TABLE row=10 col=0]
10

[P01151 | 2171:2174 | NORMAL_TEXT | TABLE row=10 col=1]
25

[P01152 | 2175:2182 | NORMAL_TEXT | TABLE row=10 col=2]
0.9511

[P01153 | 2183:2190 | NORMAL_TEXT | TABLE row=10 col=3]
0.9527

[P01154 | 2191:2198 | NORMAL_TEXT | TABLE row=10 col=4]
0.9527

[P01155 | 2199:2206 | NORMAL_TEXT | TABLE row=10 col=5]
0.9930

[P01156 | 2207:2214 | NORMAL_TEXT | TABLE row=10 col=6]
0.9528

[P01157 | 2216:2219 | NORMAL_TEXT | TABLE row=11 col=0]
11

[P01158 | 2220:2224 | NORMAL_TEXT | TABLE row=11 col=1]
228

[P01159 | 2225:2232 | NORMAL_TEXT | TABLE row=11 col=2]
0.4563

[P01160 | 2233:2240 | NORMAL_TEXT | TABLE row=11 col=3]
0.4606

[P01161 | 2241:2248 | NORMAL_TEXT | TABLE row=11 col=4]
0.2215

[P01162 | 2249:2256 | NORMAL_TEXT | TABLE row=11 col=5]
0.9375

[P01163 | 2257:2264 | NORMAL_TEXT | TABLE row=11 col=6]
0.8553

[P01164 | 2266:2269 | NORMAL_TEXT | TABLE row=12 col=0]
12

[P01165 | 2270:2274 | NORMAL_TEXT | TABLE row=12 col=1]
250

[P01166 | 2275:2282 | NORMAL_TEXT | TABLE row=12 col=2]
0.4514

[P01167 | 2283:2290 | NORMAL_TEXT | TABLE row=12 col=3]
0.4407

[P01168 | 2291:2298 | NORMAL_TEXT | TABLE row=12 col=4]
0.3377

[P01169 | 2299:2306 | NORMAL_TEXT | TABLE row=12 col=5]
0.9501

[P01170 | 2307:2314 | NORMAL_TEXT | TABLE row=12 col=6]
0.9369

[P01171 | 2316:2319 | NORMAL_TEXT | TABLE row=13 col=0]
13

[P01172 | 2320:2324 | NORMAL_TEXT | TABLE row=13 col=1]
250

[P01173 | 2325:2332 | NORMAL_TEXT | TABLE row=13 col=2]
0.1155

[P01174 | 2333:2340 | NORMAL_TEXT | TABLE row=13 col=3]
0.2064

[P01175 | 2341:2348 | NORMAL_TEXT | TABLE row=13 col=4]
0.5183

[P01176 | 2349:2356 | NORMAL_TEXT | TABLE row=13 col=5]
0.9240

[P01177 | 2357:2364 | NORMAL_TEXT | TABLE row=13 col=6]
0.8436

[P01178 | 2366:2369 | NORMAL_TEXT | TABLE row=14 col=0]
14

[P01179 | 2370:2374 | NORMAL_TEXT | TABLE row=14 col=1]
249

[P01180 | 2375:2382 | NORMAL_TEXT | TABLE row=14 col=2]
0.2626

[P01181 | 2383:2390 | NORMAL_TEXT | TABLE row=14 col=3]
0.2577

[P01182 | 2391:2398 | NORMAL_TEXT | TABLE row=14 col=4]
0.4162

[P01183 | 2399:2406 | NORMAL_TEXT | TABLE row=14 col=5]
0.8406

[P01184 | 2407:2414 | NORMAL_TEXT | TABLE row=14 col=6]
0.8239

[P01185 | 2416:2419 | NORMAL_TEXT | TABLE row=15 col=0]
15

[P01186 | 2420:2424 | NORMAL_TEXT | TABLE row=15 col=1]
116

[P01187 | 2425:2432 | NORMAL_TEXT | TABLE row=15 col=2]
0.9073

[P01188 | 2433:2440 | NORMAL_TEXT | TABLE row=15 col=3]
0.9118

[P01189 | 2441:2448 | NORMAL_TEXT | TABLE row=15 col=4]
0.8982

[P01190 | 2449:2456 | NORMAL_TEXT | TABLE row=15 col=5]
0.5822

[P01191 | 2457:2464 | NORMAL_TEXT | TABLE row=15 col=6]
0.8268

[P01192 | 2466:2469 | NORMAL_TEXT | TABLE row=16 col=0]
16

[P01193 | 2470:2473 | NORMAL_TEXT | TABLE row=16 col=1]
40

[P01194 | 2474:2481 | NORMAL_TEXT | TABLE row=16 col=2]
0.9489

[P01195 | 2482:2489 | NORMAL_TEXT | TABLE row=16 col=3]
0.9528

[P01196 | 2490:2497 | NORMAL_TEXT | TABLE row=16 col=4]
0.9536

[P01197 | 2498:2505 | NORMAL_TEXT | TABLE row=16 col=5]
0.9816

[P01198 | 2506:2513 | NORMAL_TEXT | TABLE row=16 col=6]
0.9546

[P01199 | 2515:2518 | NORMAL_TEXT | TABLE row=17 col=0]
17

[P01200 | 2519:2522 | NORMAL_TEXT | TABLE row=17 col=1]
63

[P01201 | 2523:2530 | NORMAL_TEXT | TABLE row=17 col=2]
0.8617

[P01202 | 2531:2538 | NORMAL_TEXT | TABLE row=17 col=3]
0.8633

[P01203 | 2539:2546 | NORMAL_TEXT | TABLE row=17 col=4]
0.8542

[P01204 | 2547:2554 | NORMAL_TEXT | TABLE row=17 col=5]
0.9657

[P01205 | 2555:2562 | NORMAL_TEXT | TABLE row=17 col=6]
0.9471

[P01206 | 2564:2567 | NORMAL_TEXT | TABLE row=18 col=0]
18

[P01207 | 2568:2571 | NORMAL_TEXT | TABLE row=18 col=1]
63

[P01208 | 2572:2579 | NORMAL_TEXT | TABLE row=18 col=2]
0.6229

[P01209 | 2580:2587 | NORMAL_TEXT | TABLE row=18 col=3]
0.5642

[P01210 | 2588:2595 | NORMAL_TEXT | TABLE row=18 col=4]
0.5372

[P01211 | 2596:2603 | NORMAL_TEXT | TABLE row=18 col=5]
0.8493

[P01212 | 2604:2611 | NORMAL_TEXT | TABLE row=18 col=6]
0.2884

[P01213 | 2613:2616 | NORMAL_TEXT | TABLE row=19 col=0]
19

[P01214 | 2617:2620 | NORMAL_TEXT | TABLE row=19 col=1]
56

[P01215 | 2621:2628 | NORMAL_TEXT | TABLE row=19 col=2]
0.8331

[P01216 | 2629:2636 | NORMAL_TEXT | TABLE row=19 col=3]
0.8363

[P01217 | 2637:2644 | NORMAL_TEXT | TABLE row=19 col=4]
0.8478

[P01218 | 2645:2652 | NORMAL_TEXT | TABLE row=19 col=5]
0.9475

[P01219 | 2653:2660 | NORMAL_TEXT | TABLE row=19 col=6]
0.8830

[P01220 | 2662:2665 | NORMAL_TEXT | TABLE row=20 col=0]
20

[P01221 | 2666:2669 | NORMAL_TEXT | TABLE row=20 col=1]
52

[P01222 | 2670:2677 | NORMAL_TEXT | TABLE row=20 col=2]
0.4211

[P01223 | 2678:2685 | NORMAL_TEXT | TABLE row=20 col=3]
0.5902

[P01224 | 2686:2693 | NORMAL_TEXT | TABLE row=20 col=4]
0.5678

[P01225 | 2694:2701 | NORMAL_TEXT | TABLE row=20 col=5]
0.9019

[P01226 | 2702:2709 | NORMAL_TEXT | TABLE row=20 col=6]
0.5451

[P01227 | 2711:2714 | NORMAL_TEXT | TABLE row=21 col=0]
21

[P01228 | 2715:2718 | NORMAL_TEXT | TABLE row=21 col=1]
56

[P01229 | 2719:2726 | NORMAL_TEXT | TABLE row=21 col=2]
0.7700

[P01230 | 2727:2734 | NORMAL_TEXT | TABLE row=21 col=3]
0.6996

[P01231 | 2735:2742 | NORMAL_TEXT | TABLE row=21 col=4]
0.6344

[P01232 | 2743:2750 | NORMAL_TEXT | TABLE row=21 col=5]
0.8383

[P01233 | 2751:2758 | NORMAL_TEXT | TABLE row=21 col=6]
0.9378

[P01234 | 2760:2763 | NORMAL_TEXT | TABLE row=22 col=0]
22

[P01235 | 2764:2767 | NORMAL_TEXT | TABLE row=22 col=1]
46

[P01236 | 2768:2775 | NORMAL_TEXT | TABLE row=22 col=2]
0.6853

[P01237 | 2776:2783 | NORMAL_TEXT | TABLE row=22 col=3]
0.6736

[P01238 | 2784:2791 | NORMAL_TEXT | TABLE row=22 col=4]
0.7111

[P01239 | 2792:2799 | NORMAL_TEXT | TABLE row=22 col=5]
0.9532

[P01240 | 2800:2807 | NORMAL_TEXT | TABLE row=22 col=6]
0.9273

[P01241 | 2809:2812 | NORMAL_TEXT | TABLE row=23 col=0]
23

[P01242 | 2813:2816 | NORMAL_TEXT | TABLE row=23 col=1]
56

[P01243 | 2817:2824 | NORMAL_TEXT | TABLE row=23 col=2]
0.3584

[P01244 | 2825:2832 | NORMAL_TEXT | TABLE row=23 col=3]
0.3232

[P01245 | 2833:2840 | NORMAL_TEXT | TABLE row=23 col=4]
0.3663

[P01246 | 2841:2848 | NORMAL_TEXT | TABLE row=23 col=5]
0.9514

[P01247 | 2849:2856 | NORMAL_TEXT | TABLE row=23 col=6]
0.7412

[P01248 | 2857:2858 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01249 | 2858:2859 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01250 | 2862:2870 | NORMAL_TEXT | TABLE row=0 col=0]
Pattern

[P01251 | 2871:2881 | NORMAL_TEXT | TABLE row=0 col=1]
Sequences

[P01252 | 2882:2903 | NORMAL_TEXT | TABLE row=0 col=2]
Short interpretation

[P01253 | 2905:2937 | NORMAL_TEXT | TABLE row=1 col=0]
High performance across methods

[P01254 | 2938:2948 | NORMAL_TEXT | TABLE row=1 col=1]
2, 10, 16

[P01255 | 2949:2982 | NORMAL_TEXT | TABLE row=1 col=2]
The target is consistently easy.

[P01256 | 2984:3028 | NORMAL_TEXT | TABLE row=2 col=0]
Frame-wise dataset-box method underperforms

[P01257 | 3029:3032 | NORMAL_TEXT | TABLE row=2 col=1]
15

[P01258 | 3033:3109 | NORMAL_TEXT | TABLE row=2 col=2]
One target box per frame is insufficient for multiple/disconnected regions.

[P01259 | 3111:3165 | NORMAL_TEXT | TABLE row=3 col=0]
Predicted-IoU memory substantially underperforms FIFO

[P01260 | 3166:3176 | NORMAL_TEXT | TABLE row=3 col=1]
6, 11, 12

[P01261 | 3177:3247 | NORMAL_TEXT | TABLE row=3 col=2]
High-confidence old memories can be less useful than recent memories.

[P01262 | 3249:3288 | NORMAL_TEXT | TABLE row=4 col=0]
Predicted-IoU memory gives large gains

[P01263 | 3289:3296 | NORMAL_TEXT | TABLE row=4 col=1]
13, 14

[P01264 | 3297:3364 | NORMAL_TEXT | TABLE row=4 col=2]
FIFO propagation drifts; selecting cleaner historical masks helps.

[P01265 | 3366:3402 | NORMAL_TEXT | TABLE row=5 col=0]
Small predicted-IoU-memory declines

[P01266 | 3403:3421 | NORMAL_TEXT | TABLE row=5 col=1]
8, 17, 18, 20, 21

[P01267 | 3422:3470 | NORMAL_TEXT | TABLE row=5 col=2]
Ranking does not consistently improve tracking.

[P01268 | 3471:3587 | NORMAL_TEXT]
The central conclusion is: predicted-IoU ranking is sequence-dependent, not an overall replacement for FIFO memory.

[P01269 | 3587:3626 | HEADING_2]
Does predicted-IoU-ranked memory help?

[P01270 | 3626:3627 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01271 | 3630:3638 | NORMAL_TEXT | TABLE row=0 col=0]
Measure

[P01272 | 3639:3659 | NORMAL_TEXT | TABLE row=0 col=1]
FIFO spatial memory

[P01273 | 3660:3689 | NORMAL_TEXT | TABLE row=0 col=2]
Predicted-IoU spatial memory

[P01274 | 3690:3697 | NORMAL_TEXT | TABLE row=0 col=3]
Change

[P01275 | 3699:3710 | NORMAL_TEXT | TABLE row=1 col=0]
Frame Dice

[P01276 | 3711:3718 | NORMAL_TEXT | TABLE row=1 col=1]
0.5640

[P01277 | 3719:3726 | NORMAL_TEXT | TABLE row=1 col=2]
0.5635

[P01278 | 3727:3735 | NORMAL_TEXT | TABLE row=1 col=3]
−0.0005

[P01279 | 3737:3747 | NORMAL_TEXT | TABLE row=2 col=0]
Frame IoU

[P01280 | 3748:3755 | NORMAL_TEXT | TABLE row=2 col=1]
0.5276

[P01281 | 3756:3763 | NORMAL_TEXT | TABLE row=2 col=2]
0.5139

[P01282 | 3764:3772 | NORMAL_TEXT | TABLE row=2 col=3]
−0.0138

[P01283 | 3774:3788 | NORMAL_TEXT | TABLE row=3 col=0]
Sequence Dice

[P01284 | 3789:3796 | NORMAL_TEXT | TABLE row=3 col=1]
0.7001

[P01285 | 3797:3804 | NORMAL_TEXT | TABLE row=3 col=2]
0.6864

[P01286 | 3805:3813 | NORMAL_TEXT | TABLE row=3 col=3]
−0.0137

[P01287 | 3815:3828 | NORMAL_TEXT | TABLE row=4 col=0]
Sequence IoU

[P01288 | 3829:3836 | NORMAL_TEXT | TABLE row=4 col=1]
0.6589

[P01289 | 3837:3844 | NORMAL_TEXT | TABLE row=4 col=2]
0.6399

[P01290 | 3845:3853 | NORMAL_TEXT | TABLE row=4 col=3]
−0.0191

[P01291 | 3855:3871 | NORMAL_TEXT | TABLE row=5 col=0]
Frame precision

[P01292 | 3872:3879 | NORMAL_TEXT | TABLE row=5 col=1]
0.6955

[P01293 | 3880:3887 | NORMAL_TEXT | TABLE row=5 col=2]
0.6012

[P01294 | 3888:3896 | NORMAL_TEXT | TABLE row=5 col=3]
−0.0943

[P01295 | 3897:3972 | NORMAL_TEXT | LIST id=kix.xnfoyyo12zh5 level=0]
precision: Out of all predicted positives, how many are actually positive?

[P01296 | 3972:4038 | NORMAL_TEXT | LIST id=kix.xnfoyyo12zh5 level=0]
recall: Out of all actual positives, how many did the model find?

[P01297 | 4041:4052 | NORMAL_TEXT | TABLE row=0 col=0]
Diagnostic

[P01298 | 4053:4073 | NORMAL_TEXT | TABLE row=0 col=1]
FIFO spatial memory

[P01299 | 4074:4103 | NORMAL_TEXT | TABLE row=0 col=2]
Predicted-IoU spatial memory

[P01300 | 4105:4112 | NORMAL_TEXT | TABLE row=1 col=0]
Recall

[P01301 | 4113:4120 | NORMAL_TEXT | TABLE row=1 col=1]
0.6442

[P01302 | 4121:4128 | NORMAL_TEXT | TABLE row=1 col=2]
0.7289

[P01303 | 4130:4140 | NORMAL_TEXT | TABLE row=2 col=0]
Precision

[P01304 | 4141:4148 | NORMAL_TEXT | TABLE row=2 col=1]
0.6955

[P01305 | 4149:4156 | NORMAL_TEXT | TABLE row=2 col=2]
0.6012

[P01306 | 4158:4204 | NORMAL_TEXT | TABLE row=3 col=0]
Empty predictions on 1,710 positive-GT frames

[P01307 | 4205:4209 | NORMAL_TEXT | TABLE row=3 col=1]
230

[P01308 | 4210:4213 | NORMAL_TEXT | TABLE row=3 col=2]
60

[P01309 | 4215:4261 | NORMAL_TEXT | TABLE row=4 col=0]
Foreground predictions on 515 empty-GT frames

[P01310 | 4262:4266 | NORMAL_TEXT | TABLE row=4 col=1]
191

[P01311 | 4267:4271 | NORMAL_TEXT | TABLE row=4 col=2]
255

[P01312 | 4272:4412 | NORMAL_TEXT]
Interpretation: Predicted-IoU ranking reduces complete misses, but produces more false foreground. It does not improve Dice or IoU overall.

[P01313 | 4412:4457 | NORMAL_TEXT | LIST id=kix.t1j7iinysl2t level=0]
Largest gains: seq13 +0.3627, seq14 +0.1533.

[P01314 | 4457:4502 | NORMAL_TEXT | LIST id=kix.t1j7iinysl2t level=0]
Largest losses: seq6 −0.3147, seq11 −0.2153.

[P01315 | 4502:4559 | NORMAL_TEXT | LIST id=kix.t1j7iinysl2t level=0]
All declining sequences: 2, 4, 6, 8, 11, 15, 17, 18, 20.

[P01316 | 4559:4670 | NORMAL_TEXT | LIST id=kix.t1j7iinysl2t level=0]
Excluding seq13, the frame-Dice advantage becomes −0.0129. The overall gain depends strongly on that sequence.

[P01317 | 4673:4682 | NORMAL_TEXT | TABLE row=0 col=0]
Sequence

[P01318 | 4683:4693 | NORMAL_TEXT | TABLE row=0 col=1]
FIFO Dice

[P01319 | 4694:4713 | NORMAL_TEXT | TABLE row=0 col=2]
Predicted-IoU Dice

[P01320 | 4714:4721 | NORMAL_TEXT | TABLE row=0 col=3]
Change

[P01321 | 4723:4729 | NORMAL_TEXT | TABLE row=1 col=0]
seq13

[P01322 | 4730:4737 | NORMAL_TEXT | TABLE row=1 col=1]
0.2064

[P01323 | 4738:4745 | NORMAL_TEXT | TABLE row=1 col=2]
0.5183

[P01324 | 4746:4754 | NORMAL_TEXT | TABLE row=1 col=3]
+0.3119

[P01325 | 4756:4762 | NORMAL_TEXT | TABLE row=2 col=0]
seq14

[P01326 | 4763:4770 | NORMAL_TEXT | TABLE row=2 col=1]
0.2577

[P01327 | 4771:4778 | NORMAL_TEXT | TABLE row=2 col=2]
0.4162

[P01328 | 4779:4787 | NORMAL_TEXT | TABLE row=2 col=3]
+0.1586

[P01329 | 4789:4795 | NORMAL_TEXT | TABLE row=3 col=0]
seq22

[P01330 | 4796:4803 | NORMAL_TEXT | TABLE row=3 col=1]
0.6736

[P01331 | 4804:4811 | NORMAL_TEXT | TABLE row=3 col=2]
0.7111

[P01332 | 4812:4820 | NORMAL_TEXT | TABLE row=3 col=3]
+0.0375

[P01333 | 4822:4827 | NORMAL_TEXT | TABLE row=4 col=0]
seq6

[P01334 | 4828:4835 | NORMAL_TEXT | TABLE row=4 col=1]
0.8512

[P01335 | 4836:4843 | NORMAL_TEXT | TABLE row=4 col=2]
0.4499

[P01336 | 4844:4852 | NORMAL_TEXT | TABLE row=4 col=3]
−0.4013

[P01337 | 4854:4860 | NORMAL_TEXT | TABLE row=5 col=0]
seq11

[P01338 | 4861:4868 | NORMAL_TEXT | TABLE row=5 col=1]
0.4606

[P01339 | 4869:4876 | NORMAL_TEXT | TABLE row=5 col=2]
0.2215

[P01340 | 4877:4885 | NORMAL_TEXT | TABLE row=5 col=3]
−0.2391

[P01341 | 4887:4893 | NORMAL_TEXT | TABLE row=6 col=0]
seq12

[P01342 | 4894:4901 | NORMAL_TEXT | TABLE row=6 col=1]
0.4407

[P01343 | 4902:4909 | NORMAL_TEXT | TABLE row=6 col=2]
0.3377

[P01344 | 4910:4918 | NORMAL_TEXT | TABLE row=6 col=3]
−0.1030

[P01345 | 4919:4960 | NORMAL_TEXT | LIST id=kix.qdsepfkil2tz level=0]
Gains: seq13, seq14, seq23, seq22, seq5.

[P01346 | 4960:5003 | NORMAL_TEXT | LIST id=kix.qdsepfkil2tz level=0]
Largest losses: seq6, seq11, seq12, seq21.

[P01347 | 5003:5058 | NORMAL_TEXT | LIST id=kix.qdsepfkil2tz level=0]
Declining sequences: 6, 8, 11, 12, 15, 17, 18, 20, 21.

[P01348 | 5058:5140 | NORMAL_TEXT | LIST id=kix.qdsepfkil2tz level=0]
Paired sequence-Dice change: −0.0137; 95% bootstrap interval: [-0.0667, +0.0355].

[P01349 | 5140:5153 | HEADING_2]
Failure mode

[P01350 | 5156:5169 | NORMAL_TEXT | TABLE row=0 col=0]
Failure mode

[P01351 | 5170:5179 | NORMAL_TEXT | TABLE row=0 col=1]
Evidence

[P01352 | 5180:5190 | NORMAL_TEXT | TABLE row=0 col=2]
Main idea

[P01353 | 5192:5213 | NORMAL_TEXT | TABLE row=1 col=0]
Incomplete prompting

[P01354 | 5214:5303 | NORMAL_TEXT | TABLE row=1 col=1]
Seq15 data-box frame-wise Dice: 0.9664 on one-box frames versus 0.1705 on two-box frames

[P01355 | 5304:5355 | NORMAL_TEXT | TABLE row=1 col=2]
A single box misses additional foreground regions.

[P01356 | 5357:5380 | NORMAL_TEXT | TABLE row=2 col=0]
YOLO empty predictions

[P01357 | 5381:5449 | NORMAL_TEXT | TABLE row=2 col=1]
Seq3: 13/15; seq4: 37/46; seq9: 30/42; seq18: 38/56 positive frames

[P01358 | 5450:5575 | NORMAL_TEXT | TABLE row=2 col=2]
The failure may originate in detection, segmentation, or their hand-off; detector outputs should be logged to separate them.

[P01359 | 5577:5601 | NORMAL_TEXT | TABLE row=3 col=0]
False-positive tracking

[P01360 | 5602:5687 | NORMAL_TEXT | TABLE row=3 col=1]
Seq11 predicts foreground on 90/92 empty-GT frames under both FIFO and ranked memory

[P01361 | 5688:5757 | NORMAL_TEXT | TABLE row=3 col=2]
Temporal propagation persists after the annotated object disappears.

[P01362 | 5759:5795 | NORMAL_TEXT | TABLE row=4 col=0]
Excessive/wrong-region segmentation

[P01363 | 5796:5834 | NORMAL_TEXT | TABLE row=4 col=1]
Seq11 ranked-memory precision: 0.1960

[P01364 | 5835:5899 | NORMAL_TEXT | TABLE row=4 col=2]
Ranked spatial memory frequently predicts incorrect foreground.

[P01365 | 5901:5930 | NORMAL_TEXT | TABLE row=5 col=0]
Inconsistent ranking benefit

[P01366 | 5931:5996 | NORMAL_TEXT | TABLE row=5 col=1]
Positive-frame Dice: seq13 0.0533 → 0.5657; seq6 0.8462 → 0.2732

[P01367 | 5997:6101 | NORMAL_TEXT | TABLE row=5 col=2]
Predicted IoU sometimes recovers tracking, but can also select stale or confidently incorrect memories.

[P01368 | 6102:6103 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

## Bản sao của Experiment Results (t.gxm8ux5m5rs4)

[P01369 | 1:13 | HEADING_2]
Experiments

[P01370 | 13:15 | NORMAL_TEXT]
[INLINE_OBJECT kix.44ckh8lwzh3m]

[P01371 | 15:26 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=0]
Frame-only

[P01372 | 26:48 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=1]
manual box generation

[P01373 | 48:69 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=1]
YOLv8 box generation

[P01374 | 69:85 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=0]
default MedSAM2

[P01375 | 85:135 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=1]
add first positive box prompt + naive memory bank

[P01376 | 135:208 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=1]
anchor mask + naive memory bank (object-pointer from most recent frames)

[P01377 | 208:290 | NORMAL_TEXT | LIST id=kix.rrb64b90xwj5 level=1]
anchor mask + IoU-confidence memory bank (object-pointer from most recent frames)

[P01378 | 290:306 | HEADING_2]
Overall Results

[P01379 | 309:316 | NORMAL_TEXT | TABLE row=0 col=0]
Method

[P01380 | 317:328 | NORMAL_TEXT | TABLE row=0 col=1]
Frame Dice

[P01381 | 329:339 | NORMAL_TEXT | TABLE row=0 col=2]
Frame IoU

[P01382 | 340:354 | NORMAL_TEXT | TABLE row=0 col=3]
Sequence Dice

[P01383 | 355:368 | NORMAL_TEXT | TABLE row=0 col=4]
Sequence IoU

[P01384 | 370:413 | NORMAL_TEXT | TABLE row=1 col=0]
Independent frames — dataset box per frame

[P01385 | 414:421 | NORMAL_TEXT | TABLE row=1 col=1]
0.9096

[P01386 | 422:429 | NORMAL_TEXT | TABLE row=1 col=2]
0.8658

[P01387 | 430:437 | NORMAL_TEXT | TABLE row=1 col=3]
0.9232

[P01388 | 438:445 | NORMAL_TEXT | TABLE row=1 col=4]
0.8834

[P01389 | 447:489 | NORMAL_TEXT | TABLE row=2 col=0]
Independent frames — YOLOv8 box per frame

[P01390 | 490:497 | NORMAL_TEXT | TABLE row=2 col=1]
0.8300

[P01391 | 498:505 | NORMAL_TEXT | TABLE row=2 col=2]
0.7915

[P01392 | 506:513 | NORMAL_TEXT | TABLE row=2 col=3]
0.7700

[P01393 | 514:521 | NORMAL_TEXT | TABLE row=2 col=4]
0.7392

[P01394 | 523:574 | NORMAL_TEXT | TABLE row=3 col=0]
VOS — first available dataset box, standard memory

[P01395 | 575:582 | NORMAL_TEXT | TABLE row=3 col=1]
0.5551

[P01396 | 583:590 | NORMAL_TEXT | TABLE row=3 col=2]
0.5178

[P01397 | 591:598 | NORMAL_TEXT | TABLE row=3 col=3]
0.6955

[P01398 | 599:606 | NORMAL_TEXT | TABLE row=3 col=4]
0.6535

[P01399 | 608:659 | NORMAL_TEXT | TABLE row=4 col=0]
VOS-per-object — anchor mask + FIFO spatial memory

[P01400 | 660:667 | NORMAL_TEXT | TABLE row=4 col=1]
0.5640

[P01401 | 668:675 | NORMAL_TEXT | TABLE row=4 col=2]
0.5276

[P01402 | 676:683 | NORMAL_TEXT | TABLE row=4 col=3]
0.7001

[P01403 | 684:691 | NORMAL_TEXT | TABLE row=4 col=4]
0.6589

[P01404 | 693:760 | NORMAL_TEXT | TABLE row=5 col=0]
VOS-per-object — anchor mask + predicted-IoU-ranked spatial memory

[P01405 | 761:768 | NORMAL_TEXT | TABLE row=5 col=1]
0.5635

[P01406 | 769:776 | NORMAL_TEXT | TABLE row=5 col=2]
0.5139

[P01407 | 777:784 | NORMAL_TEXT | TABLE row=5 col=3]
0.6864

[P01408 | 785:792 | NORMAL_TEXT | TABLE row=5 col=4]
0.6399

[P01409 | 793:809 | NORMAL_TEXT]
Interpretation:

[P01410 | 809:879 | NORMAL_TEXT | LIST id=kix.x8ea39te4ayf level=0]
The reported Dice/IoU differences are real differences in predictions

[P01411 | 879:919 | NORMAL_TEXT | LIST id=kix.26lszn25qf0 level=0]
Fresh localization helps substantially.

[P01412 | 919:920 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01413 | 923:934 | NORMAL_TEXT | TABLE row=0 col=0]
Frame Dice

[P01414 | 935:947 | NORMAL_TEXT | TABLE row=0 col=1]
Initial box

[P01415 | 948:960 | NORMAL_TEXT | TABLE row=0 col=2]
Anchor mask

[P01416 | 961:972 | NORMAL_TEXT | TABLE row=0 col=3]
Difference

[P01417 | 974:985 | NORMAL_TEXT | TABLE row=1 col=0]
All frames

[P01418 | 986:993 | NORMAL_TEXT | TABLE row=1 col=1]
0.5551

[P01419 | 994:1001 | NORMAL_TEXT | TABLE row=1 col=2]
0.5640

[P01420 | 1002:1010 | NORMAL_TEXT | TABLE row=1 col=3]
+0.0089

[P01421 | 1012:1044 | NORMAL_TEXT | TABLE row=2 col=0]
Excluding initialization frames

[P01422 | 1045:1052 | NORMAL_TEXT | TABLE row=2 col=1]
0.5515

[P01423 | 1053:1060 | NORMAL_TEXT | TABLE row=2 col=2]
0.5599

[P01424 | 1061:1069 | NORMAL_TEXT | TABLE row=2 col=3]
+0.0084

[P01425 | 1070:1071 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01426 | 1071:1072 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01427 | 1075:1086 | NORMAL_TEXT | TABLE row=0 col=0]
Hypothesis

[P01428 | 1087:1096 | NORMAL_TEXT | TABLE row=0 col=1]
Evidence

[P01429 | 1097:1108 | NORMAL_TEXT | TABLE row=0 col=2]
Assessment

[P01430 | 1110:1177 | NORMAL_TEXT | TABLE row=1 col=0]
Anchor mask + FIFO memory is better than initial box + FIFO memory

[P01431 | 1178:1238 | NORMAL_TEXT | TABLE row=1 col=1]
Frame Dice: 0.5551 → 0.5640; Sequence Dice: 0.6955 → 0.7001

[P01432 | 1239:1266 | NORMAL_TEXT | TABLE row=1 col=2]
Small average improvement.

[P01433 | 1268:1323 | NORMAL_TEXT | TABLE row=2 col=0]
Predicted-IoU-ranked memory is better than FIFO memory

[P01434 | 1324:1384 | NORMAL_TEXT | TABLE row=2 col=1]
Frame Dice: 0.5640 → 0.5635; Sequence Dice: 0.7001 → 0.6864

[P01435 | 1385:1408 | NORMAL_TEXT | TABLE row=2 col=2]
Not supported overall.

[P01436 | 1410:1467 | NORMAL_TEXT | TABLE row=3 col=0]
Frame-wise inference is better than temporal propagation

[P01437 | 1468:1547 | NORMAL_TEXT | TABLE row=3 col=1]
Dataset-box frame-wise method wins in 20 sequences, loses in 1, and ties in 2.

[P01438 | 1548:1620 | NORMAL_TEXT | TABLE row=3 col=2]
True for these experiments, but it receives a fresh box on every frame.

[P01439 | 1621:1622 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01440 | 1622:1623 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01441 | 1626:1630 | NORMAL_TEXT | TABLE row=0 col=0]
Seq

[P01442 | 1631:1637 | NORMAL_TEXT | TABLE row=0 col=1]
Frame

[P01443 | 1638:1652 | NORMAL_TEXT | TABLE row=0 col=2]
First-box VOS

[P01444 | 1653:1665 | NORMAL_TEXT | TABLE row=0 col=3]
Anchor FIFO

[P01445 | 1666:1687 | NORMAL_TEXT | TABLE row=0 col=4]
Anchor predicted-IoU

[P01446 | 1688:1711 | NORMAL_TEXT | TABLE row=0 col=5]
Frame-wise dataset box

[P01447 | 1712:1732 | NORMAL_TEXT | TABLE row=0 col=6]
Frame-wise YOLO box

[P01448 | 1734:1736 | NORMAL_TEXT | TABLE row=1 col=0]
1

[P01449 | 1737:1740 | NORMAL_TEXT | TABLE row=1 col=1]
36

[P01450 | 1741:1748 | NORMAL_TEXT | TABLE row=1 col=2]
1.0000

[P01451 | 1749:1756 | NORMAL_TEXT | TABLE row=1 col=3]
1.0000

[P01452 | 1757:1764 | NORMAL_TEXT | TABLE row=1 col=4]
1.0000

[P01453 | 1765:1772 | NORMAL_TEXT | TABLE row=1 col=5]
1.0000

[P01454 | 1773:1780 | NORMAL_TEXT | TABLE row=1 col=6]
1.0000

[P01455 | 1782:1784 | NORMAL_TEXT | TABLE row=2 col=0]
2

[P01456 | 1785:1788 | NORMAL_TEXT | TABLE row=2 col=1]
63

[P01457 | 1789:1796 | NORMAL_TEXT | TABLE row=2 col=2]
0.9014

[P01458 | 1797:1804 | NORMAL_TEXT | TABLE row=2 col=3]
0.9017

[P01459 | 1805:1812 | NORMAL_TEXT | TABLE row=2 col=4]
0.9023

[P01460 | 1813:1820 | NORMAL_TEXT | TABLE row=2 col=5]
0.9609

[P01461 | 1821:1828 | NORMAL_TEXT | TABLE row=2 col=6]
0.9146

[P01462 | 1830:1832 | NORMAL_TEXT | TABLE row=3 col=0]
3

[P01463 | 1833:1836 | NORMAL_TEXT | TABLE row=3 col=1]
15

[P01464 | 1837:1844 | NORMAL_TEXT | TABLE row=3 col=2]
0.8381

[P01465 | 1845:1852 | NORMAL_TEXT | TABLE row=3 col=3]
0.8483

[P01466 | 1853:1860 | NORMAL_TEXT | TABLE row=3 col=4]
0.8594

[P01467 | 1861:1868 | NORMAL_TEXT | TABLE row=3 col=5]
0.9424

[P01468 | 1869:1876 | NORMAL_TEXT | TABLE row=3 col=6]
0.0614

[P01469 | 1878:1880 | NORMAL_TEXT | TABLE row=4 col=0]
4

[P01470 | 1881:1884 | NORMAL_TEXT | TABLE row=4 col=1]
48

[P01471 | 1885:1892 | NORMAL_TEXT | TABLE row=4 col=2]
0.7886

[P01472 | 1893:1900 | NORMAL_TEXT | TABLE row=4 col=3]
0.7909

[P01473 | 1901:1908 | NORMAL_TEXT | TABLE row=4 col=4]
0.7913

[P01474 | 1909:1916 | NORMAL_TEXT | TABLE row=4 col=5]
0.9511

[P01475 | 1917:1924 | NORMAL_TEXT | TABLE row=4 col=6]
0.1673

[P01476 | 1926:1928 | NORMAL_TEXT | TABLE row=5 col=0]
5

[P01477 | 1929:1933 | NORMAL_TEXT | TABLE row=5 col=1]
250

[P01478 | 1934:1941 | NORMAL_TEXT | TABLE row=5 col=2]
0.5867

[P01479 | 1942:1949 | NORMAL_TEXT | TABLE row=5 col=3]
0.5828

[P01480 | 1950:1957 | NORMAL_TEXT | TABLE row=5 col=4]
0.5980

[P01481 | 1958:1965 | NORMAL_TEXT | TABLE row=5 col=5]
0.9420

[P01482 | 1966:1973 | NORMAL_TEXT | TABLE row=5 col=6]
0.9302

[P01483 | 1975:1977 | NORMAL_TEXT | TABLE row=6 col=0]
6

[P01484 | 1978:1981 | NORMAL_TEXT | TABLE row=6 col=1]
91

[P01485 | 1982:1989 | NORMAL_TEXT | TABLE row=6 col=2]
0.8443

[P01486 | 1990:1997 | NORMAL_TEXT | TABLE row=6 col=3]
0.8512

[P01487 | 1998:2005 | NORMAL_TEXT | TABLE row=6 col=4]
0.4499

[P01488 | 2006:2013 | NORMAL_TEXT | TABLE row=6 col=5]
0.9606

[P01489 | 2014:2021 | NORMAL_TEXT | TABLE row=6 col=6]
0.9451

[P01490 | 2023:2025 | NORMAL_TEXT | TABLE row=7 col=0]
7

[P01491 | 2026:2029 | NORMAL_TEXT | TABLE row=7 col=1]
48

[P01492 | 2030:2037 | NORMAL_TEXT | TABLE row=7 col=2]
1.0000

[P01493 | 2038:2045 | NORMAL_TEXT | TABLE row=7 col=3]
1.0000

[P01494 | 2046:2053 | NORMAL_TEXT | TABLE row=7 col=4]
1.0000

[P01495 | 2054:2061 | NORMAL_TEXT | TABLE row=7 col=5]
1.0000

[P01496 | 2062:2069 | NORMAL_TEXT | TABLE row=7 col=6]
1.0000

[P01497 | 2071:2073 | NORMAL_TEXT | TABLE row=8 col=0]
8

[P01498 | 2074:2077 | NORMAL_TEXT | TABLE row=8 col=1]
73

[P01499 | 2078:2085 | NORMAL_TEXT | TABLE row=8 col=2]
0.6557

[P01500 | 2086:2093 | NORMAL_TEXT | TABLE row=8 col=3]
0.6685

[P01501 | 2094:2101 | NORMAL_TEXT | TABLE row=8 col=4]
0.6399

[P01502 | 2102:2109 | NORMAL_TEXT | TABLE row=8 col=5]
0.9181

[P01503 | 2110:2117 | NORMAL_TEXT | TABLE row=8 col=6]
0.8875

[P01504 | 2119:2121 | NORMAL_TEXT | TABLE row=9 col=0]
9

[P01505 | 2122:2125 | NORMAL_TEXT | TABLE row=9 col=1]
51

[P01506 | 2126:2133 | NORMAL_TEXT | TABLE row=9 col=2]
0.7356

[P01507 | 2134:2141 | NORMAL_TEXT | TABLE row=9 col=3]
0.7258

[P01508 | 2142:2149 | NORMAL_TEXT | TABLE row=9 col=4]
0.7295

[P01509 | 2150:2157 | NORMAL_TEXT | TABLE row=9 col=5]
0.9419

[P01510 | 2158:2165 | NORMAL_TEXT | TABLE row=9 col=6]
0.3401

[P01511 | 2167:2170 | NORMAL_TEXT | TABLE row=10 col=0]
10

[P01512 | 2171:2174 | NORMAL_TEXT | TABLE row=10 col=1]
25

[P01513 | 2175:2182 | NORMAL_TEXT | TABLE row=10 col=2]
0.9511

[P01514 | 2183:2190 | NORMAL_TEXT | TABLE row=10 col=3]
0.9527

[P01515 | 2191:2198 | NORMAL_TEXT | TABLE row=10 col=4]
0.9527

[P01516 | 2199:2206 | NORMAL_TEXT | TABLE row=10 col=5]
0.9930

[P01517 | 2207:2214 | NORMAL_TEXT | TABLE row=10 col=6]
0.9528

[P01518 | 2216:2219 | NORMAL_TEXT | TABLE row=11 col=0]
11

[P01519 | 2220:2224 | NORMAL_TEXT | TABLE row=11 col=1]
228

[P01520 | 2225:2232 | NORMAL_TEXT | TABLE row=11 col=2]
0.4563

[P01521 | 2233:2240 | NORMAL_TEXT | TABLE row=11 col=3]
0.4606

[P01522 | 2241:2248 | NORMAL_TEXT | TABLE row=11 col=4]
0.2215

[P01523 | 2249:2256 | NORMAL_TEXT | TABLE row=11 col=5]
0.9375

[P01524 | 2257:2264 | NORMAL_TEXT | TABLE row=11 col=6]
0.8553

[P01525 | 2266:2269 | NORMAL_TEXT | TABLE row=12 col=0]
12

[P01526 | 2270:2274 | NORMAL_TEXT | TABLE row=12 col=1]
250

[P01527 | 2275:2282 | NORMAL_TEXT | TABLE row=12 col=2]
0.4514

[P01528 | 2283:2290 | NORMAL_TEXT | TABLE row=12 col=3]
0.4407

[P01529 | 2291:2298 | NORMAL_TEXT | TABLE row=12 col=4]
0.3377

[P01530 | 2299:2306 | NORMAL_TEXT | TABLE row=12 col=5]
0.9501

[P01531 | 2307:2314 | NORMAL_TEXT | TABLE row=12 col=6]
0.9369

[P01532 | 2316:2319 | NORMAL_TEXT | TABLE row=13 col=0]
13

[P01533 | 2320:2324 | NORMAL_TEXT | TABLE row=13 col=1]
250

[P01534 | 2325:2332 | NORMAL_TEXT | TABLE row=13 col=2]
0.1155

[P01535 | 2333:2340 | NORMAL_TEXT | TABLE row=13 col=3]
0.2064

[P01536 | 2341:2348 | NORMAL_TEXT | TABLE row=13 col=4]
0.5183

[P01537 | 2349:2356 | NORMAL_TEXT | TABLE row=13 col=5]
0.9240

[P01538 | 2357:2364 | NORMAL_TEXT | TABLE row=13 col=6]
0.8436

[P01539 | 2366:2369 | NORMAL_TEXT | TABLE row=14 col=0]
14

[P01540 | 2370:2374 | NORMAL_TEXT | TABLE row=14 col=1]
249

[P01541 | 2375:2382 | NORMAL_TEXT | TABLE row=14 col=2]
0.2626

[P01542 | 2383:2390 | NORMAL_TEXT | TABLE row=14 col=3]
0.2577

[P01543 | 2391:2398 | NORMAL_TEXT | TABLE row=14 col=4]
0.4162

[P01544 | 2399:2406 | NORMAL_TEXT | TABLE row=14 col=5]
0.8406

[P01545 | 2407:2414 | NORMAL_TEXT | TABLE row=14 col=6]
0.8239

[P01546 | 2416:2419 | NORMAL_TEXT | TABLE row=15 col=0]
15

[P01547 | 2420:2424 | NORMAL_TEXT | TABLE row=15 col=1]
116

[P01548 | 2425:2432 | NORMAL_TEXT | TABLE row=15 col=2]
0.9073

[P01549 | 2433:2440 | NORMAL_TEXT | TABLE row=15 col=3]
0.9118

[P01550 | 2441:2448 | NORMAL_TEXT | TABLE row=15 col=4]
0.8982

[P01551 | 2449:2456 | NORMAL_TEXT | TABLE row=15 col=5]
0.5822

[P01552 | 2457:2464 | NORMAL_TEXT | TABLE row=15 col=6]
0.8268

[P01553 | 2466:2469 | NORMAL_TEXT | TABLE row=16 col=0]
16

[P01554 | 2470:2473 | NORMAL_TEXT | TABLE row=16 col=1]
40

[P01555 | 2474:2481 | NORMAL_TEXT | TABLE row=16 col=2]
0.9489

[P01556 | 2482:2489 | NORMAL_TEXT | TABLE row=16 col=3]
0.9528

[P01557 | 2490:2497 | NORMAL_TEXT | TABLE row=16 col=4]
0.9536

[P01558 | 2498:2505 | NORMAL_TEXT | TABLE row=16 col=5]
0.9816

[P01559 | 2506:2513 | NORMAL_TEXT | TABLE row=16 col=6]
0.9546

[P01560 | 2515:2518 | NORMAL_TEXT | TABLE row=17 col=0]
17

[P01561 | 2519:2522 | NORMAL_TEXT | TABLE row=17 col=1]
63

[P01562 | 2523:2530 | NORMAL_TEXT | TABLE row=17 col=2]
0.8617

[P01563 | 2531:2538 | NORMAL_TEXT | TABLE row=17 col=3]
0.8633

[P01564 | 2539:2546 | NORMAL_TEXT | TABLE row=17 col=4]
0.8542

[P01565 | 2547:2554 | NORMAL_TEXT | TABLE row=17 col=5]
0.9657

[P01566 | 2555:2562 | NORMAL_TEXT | TABLE row=17 col=6]
0.9471

[P01567 | 2564:2567 | NORMAL_TEXT | TABLE row=18 col=0]
18

[P01568 | 2568:2571 | NORMAL_TEXT | TABLE row=18 col=1]
63

[P01569 | 2572:2579 | NORMAL_TEXT | TABLE row=18 col=2]
0.6229

[P01570 | 2580:2587 | NORMAL_TEXT | TABLE row=18 col=3]
0.5642

[P01571 | 2588:2595 | NORMAL_TEXT | TABLE row=18 col=4]
0.5372

[P01572 | 2596:2603 | NORMAL_TEXT | TABLE row=18 col=5]
0.8493

[P01573 | 2604:2611 | NORMAL_TEXT | TABLE row=18 col=6]
0.2884

[P01574 | 2613:2616 | NORMAL_TEXT | TABLE row=19 col=0]
19

[P01575 | 2617:2620 | NORMAL_TEXT | TABLE row=19 col=1]
56

[P01576 | 2621:2628 | NORMAL_TEXT | TABLE row=19 col=2]
0.8331

[P01577 | 2629:2636 | NORMAL_TEXT | TABLE row=19 col=3]
0.8363

[P01578 | 2637:2644 | NORMAL_TEXT | TABLE row=19 col=4]
0.8478

[P01579 | 2645:2652 | NORMAL_TEXT | TABLE row=19 col=5]
0.9475

[P01580 | 2653:2660 | NORMAL_TEXT | TABLE row=19 col=6]
0.8830

[P01581 | 2662:2665 | NORMAL_TEXT | TABLE row=20 col=0]
20

[P01582 | 2666:2669 | NORMAL_TEXT | TABLE row=20 col=1]
52

[P01583 | 2670:2677 | NORMAL_TEXT | TABLE row=20 col=2]
0.4211

[P01584 | 2678:2685 | NORMAL_TEXT | TABLE row=20 col=3]
0.5902

[P01585 | 2686:2693 | NORMAL_TEXT | TABLE row=20 col=4]
0.5678

[P01586 | 2694:2701 | NORMAL_TEXT | TABLE row=20 col=5]
0.9019

[P01587 | 2702:2709 | NORMAL_TEXT | TABLE row=20 col=6]
0.5451

[P01588 | 2711:2714 | NORMAL_TEXT | TABLE row=21 col=0]
21

[P01589 | 2715:2718 | NORMAL_TEXT | TABLE row=21 col=1]
56

[P01590 | 2719:2726 | NORMAL_TEXT | TABLE row=21 col=2]
0.7700

[P01591 | 2727:2734 | NORMAL_TEXT | TABLE row=21 col=3]
0.6996

[P01592 | 2735:2742 | NORMAL_TEXT | TABLE row=21 col=4]
0.6344

[P01593 | 2743:2750 | NORMAL_TEXT | TABLE row=21 col=5]
0.8383

[P01594 | 2751:2758 | NORMAL_TEXT | TABLE row=21 col=6]
0.9378

[P01595 | 2760:2763 | NORMAL_TEXT | TABLE row=22 col=0]
22

[P01596 | 2764:2767 | NORMAL_TEXT | TABLE row=22 col=1]
46

[P01597 | 2768:2775 | NORMAL_TEXT | TABLE row=22 col=2]
0.6853

[P01598 | 2776:2783 | NORMAL_TEXT | TABLE row=22 col=3]
0.6736

[P01599 | 2784:2791 | NORMAL_TEXT | TABLE row=22 col=4]
0.7111

[P01600 | 2792:2799 | NORMAL_TEXT | TABLE row=22 col=5]
0.9532

[P01601 | 2800:2807 | NORMAL_TEXT | TABLE row=22 col=6]
0.9273

[P01602 | 2809:2812 | NORMAL_TEXT | TABLE row=23 col=0]
23

[P01603 | 2813:2816 | NORMAL_TEXT | TABLE row=23 col=1]
56

[P01604 | 2817:2824 | NORMAL_TEXT | TABLE row=23 col=2]
0.3584

[P01605 | 2825:2832 | NORMAL_TEXT | TABLE row=23 col=3]
0.3232

[P01606 | 2833:2840 | NORMAL_TEXT | TABLE row=23 col=4]
0.3663

[P01607 | 2841:2848 | NORMAL_TEXT | TABLE row=23 col=5]
0.9514

[P01608 | 2849:2856 | NORMAL_TEXT | TABLE row=23 col=6]
0.7412

[P01609 | 2857:2858 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01610 | 2858:2859 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01611 | 2862:2870 | NORMAL_TEXT | TABLE row=0 col=0]
Pattern

[P01612 | 2871:2881 | NORMAL_TEXT | TABLE row=0 col=1]
Sequences

[P01613 | 2882:2903 | NORMAL_TEXT | TABLE row=0 col=2]
Short interpretation

[P01614 | 2905:2937 | NORMAL_TEXT | TABLE row=1 col=0]
High performance across methods

[P01615 | 2938:2948 | NORMAL_TEXT | TABLE row=1 col=1]
2, 10, 16

[P01616 | 2949:2982 | NORMAL_TEXT | TABLE row=1 col=2]
The target is consistently easy.

[P01617 | 2984:3028 | NORMAL_TEXT | TABLE row=2 col=0]
Frame-wise dataset-box method underperforms

[P01618 | 3029:3032 | NORMAL_TEXT | TABLE row=2 col=1]
15

[P01619 | 3033:3109 | NORMAL_TEXT | TABLE row=2 col=2]
One target box per frame is insufficient for multiple/disconnected regions.

[P01620 | 3111:3165 | NORMAL_TEXT | TABLE row=3 col=0]
Predicted-IoU memory substantially underperforms FIFO

[P01621 | 3166:3176 | NORMAL_TEXT | TABLE row=3 col=1]
6, 11, 12

[P01622 | 3177:3247 | NORMAL_TEXT | TABLE row=3 col=2]
High-confidence old memories can be less useful than recent memories.

[P01623 | 3249:3288 | NORMAL_TEXT | TABLE row=4 col=0]
Predicted-IoU memory gives large gains

[P01624 | 3289:3296 | NORMAL_TEXT | TABLE row=4 col=1]
13, 14

[P01625 | 3297:3364 | NORMAL_TEXT | TABLE row=4 col=2]
FIFO propagation drifts; selecting cleaner historical masks helps.

[P01626 | 3366:3402 | NORMAL_TEXT | TABLE row=5 col=0]
Small predicted-IoU-memory declines

[P01627 | 3403:3421 | NORMAL_TEXT | TABLE row=5 col=1]
8, 17, 18, 20, 21

[P01628 | 3422:3470 | NORMAL_TEXT | TABLE row=5 col=2]
Ranking does not consistently improve tracking.

[P01629 | 3471:3587 | NORMAL_TEXT]
The central conclusion is: predicted-IoU ranking is sequence-dependent, not an overall replacement for FIFO memory.

[P01630 | 3587:3626 | HEADING_2]
Does predicted-IoU-ranked memory help?

[P01631 | 3626:3627 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

[P01632 | 3630:3638 | NORMAL_TEXT | TABLE row=0 col=0]
Measure

[P01633 | 3639:3659 | NORMAL_TEXT | TABLE row=0 col=1]
FIFO spatial memory

[P01634 | 3660:3689 | NORMAL_TEXT | TABLE row=0 col=2]
Predicted-IoU spatial memory

[P01635 | 3690:3697 | NORMAL_TEXT | TABLE row=0 col=3]
Change

[P01636 | 3699:3710 | NORMAL_TEXT | TABLE row=1 col=0]
Frame Dice

[P01637 | 3711:3718 | NORMAL_TEXT | TABLE row=1 col=1]
0.5640

[P01638 | 3719:3726 | NORMAL_TEXT | TABLE row=1 col=2]
0.5635

[P01639 | 3727:3735 | NORMAL_TEXT | TABLE row=1 col=3]
−0.0005

[P01640 | 3737:3747 | NORMAL_TEXT | TABLE row=2 col=0]
Frame IoU

[P01641 | 3748:3755 | NORMAL_TEXT | TABLE row=2 col=1]
0.5276

[P01642 | 3756:3763 | NORMAL_TEXT | TABLE row=2 col=2]
0.5139

[P01643 | 3764:3772 | NORMAL_TEXT | TABLE row=2 col=3]
−0.0138

[P01644 | 3774:3788 | NORMAL_TEXT | TABLE row=3 col=0]
Sequence Dice

[P01645 | 3789:3796 | NORMAL_TEXT | TABLE row=3 col=1]
0.7001

[P01646 | 3797:3804 | NORMAL_TEXT | TABLE row=3 col=2]
0.6864

[P01647 | 3805:3813 | NORMAL_TEXT | TABLE row=3 col=3]
−0.0137

[P01648 | 3815:3828 | NORMAL_TEXT | TABLE row=4 col=0]
Sequence IoU

[P01649 | 3829:3836 | NORMAL_TEXT | TABLE row=4 col=1]
0.6589

[P01650 | 3837:3844 | NORMAL_TEXT | TABLE row=4 col=2]
0.6399

[P01651 | 3845:3853 | NORMAL_TEXT | TABLE row=4 col=3]
−0.0191

[P01652 | 3855:3871 | NORMAL_TEXT | TABLE row=5 col=0]
Frame precision

[P01653 | 3872:3879 | NORMAL_TEXT | TABLE row=5 col=1]
0.6955

[P01654 | 3880:3887 | NORMAL_TEXT | TABLE row=5 col=2]
0.6012

[P01655 | 3888:3896 | NORMAL_TEXT | TABLE row=5 col=3]
−0.0943

[P01656 | 3897:3972 | NORMAL_TEXT | LIST id=kix.xnfoyyo12zh5 level=0]
precision: Out of all predicted positives, how many are actually positive?

[P01657 | 3972:4038 | NORMAL_TEXT | LIST id=kix.xnfoyyo12zh5 level=0]
recall: Out of all actual positives, how many did the model find?

[P01658 | 4041:4052 | NORMAL_TEXT | TABLE row=0 col=0]
Diagnostic

[P01659 | 4053:4073 | NORMAL_TEXT | TABLE row=0 col=1]
FIFO spatial memory

[P01660 | 4074:4103 | NORMAL_TEXT | TABLE row=0 col=2]
Predicted-IoU spatial memory

[P01661 | 4105:4112 | NORMAL_TEXT | TABLE row=1 col=0]
Recall

[P01662 | 4113:4120 | NORMAL_TEXT | TABLE row=1 col=1]
0.6442

[P01663 | 4121:4128 | NORMAL_TEXT | TABLE row=1 col=2]
0.7289

[P01664 | 4130:4140 | NORMAL_TEXT | TABLE row=2 col=0]
Precision

[P01665 | 4141:4148 | NORMAL_TEXT | TABLE row=2 col=1]
0.6955

[P01666 | 4149:4156 | NORMAL_TEXT | TABLE row=2 col=2]
0.6012

[P01667 | 4158:4204 | NORMAL_TEXT | TABLE row=3 col=0]
Empty predictions on 1,710 positive-GT frames

[P01668 | 4205:4209 | NORMAL_TEXT | TABLE row=3 col=1]
230

[P01669 | 4210:4213 | NORMAL_TEXT | TABLE row=3 col=2]
60

[P01670 | 4215:4261 | NORMAL_TEXT | TABLE row=4 col=0]
Foreground predictions on 515 empty-GT frames

[P01671 | 4262:4266 | NORMAL_TEXT | TABLE row=4 col=1]
191

[P01672 | 4267:4271 | NORMAL_TEXT | TABLE row=4 col=2]
255

[P01673 | 4272:4412 | NORMAL_TEXT]
Interpretation: Predicted-IoU ranking reduces complete misses, but produces more false foreground. It does not improve Dice or IoU overall.

[P01674 | 4412:4457 | NORMAL_TEXT | LIST id=kix.t1j7iinysl2t level=0]
Largest gains: seq13 +0.3627, seq14 +0.1533.

[P01675 | 4457:4502 | NORMAL_TEXT | LIST id=kix.t1j7iinysl2t level=0]
Largest losses: seq6 −0.3147, seq11 −0.2153.

[P01676 | 4502:4559 | NORMAL_TEXT | LIST id=kix.t1j7iinysl2t level=0]
All declining sequences: 2, 4, 6, 8, 11, 15, 17, 18, 20.

[P01677 | 4559:4670 | NORMAL_TEXT | LIST id=kix.t1j7iinysl2t level=0]
Excluding seq13, the frame-Dice advantage becomes −0.0129. The overall gain depends strongly on that sequence.

[P01678 | 4673:4682 | NORMAL_TEXT | TABLE row=0 col=0]
Sequence

[P01679 | 4683:4693 | NORMAL_TEXT | TABLE row=0 col=1]
FIFO Dice

[P01680 | 4694:4713 | NORMAL_TEXT | TABLE row=0 col=2]
Predicted-IoU Dice

[P01681 | 4714:4721 | NORMAL_TEXT | TABLE row=0 col=3]
Change

[P01682 | 4723:4729 | NORMAL_TEXT | TABLE row=1 col=0]
seq13

[P01683 | 4730:4737 | NORMAL_TEXT | TABLE row=1 col=1]
0.2064

[P01684 | 4738:4745 | NORMAL_TEXT | TABLE row=1 col=2]
0.5183

[P01685 | 4746:4754 | NORMAL_TEXT | TABLE row=1 col=3]
+0.3119

[P01686 | 4756:4762 | NORMAL_TEXT | TABLE row=2 col=0]
seq14

[P01687 | 4763:4770 | NORMAL_TEXT | TABLE row=2 col=1]
0.2577

[P01688 | 4771:4778 | NORMAL_TEXT | TABLE row=2 col=2]
0.4162

[P01689 | 4779:4787 | NORMAL_TEXT | TABLE row=2 col=3]
+0.1586

[P01690 | 4789:4795 | NORMAL_TEXT | TABLE row=3 col=0]
seq22

[P01691 | 4796:4803 | NORMAL_TEXT | TABLE row=3 col=1]
0.6736

[P01692 | 4804:4811 | NORMAL_TEXT | TABLE row=3 col=2]
0.7111

[P01693 | 4812:4820 | NORMAL_TEXT | TABLE row=3 col=3]
+0.0375

[P01694 | 4822:4827 | NORMAL_TEXT | TABLE row=4 col=0]
seq6

[P01695 | 4828:4835 | NORMAL_TEXT | TABLE row=4 col=1]
0.8512

[P01696 | 4836:4843 | NORMAL_TEXT | TABLE row=4 col=2]
0.4499

[P01697 | 4844:4852 | NORMAL_TEXT | TABLE row=4 col=3]
−0.4013

[P01698 | 4854:4860 | NORMAL_TEXT | TABLE row=5 col=0]
seq11

[P01699 | 4861:4868 | NORMAL_TEXT | TABLE row=5 col=1]
0.4606

[P01700 | 4869:4876 | NORMAL_TEXT | TABLE row=5 col=2]
0.2215

[P01701 | 4877:4885 | NORMAL_TEXT | TABLE row=5 col=3]
−0.2391

[P01702 | 4887:4893 | NORMAL_TEXT | TABLE row=6 col=0]
seq12

[P01703 | 4894:4901 | NORMAL_TEXT | TABLE row=6 col=1]
0.4407

[P01704 | 4902:4909 | NORMAL_TEXT | TABLE row=6 col=2]
0.3377

[P01705 | 4910:4918 | NORMAL_TEXT | TABLE row=6 col=3]
−0.1030

[P01706 | 4919:4960 | NORMAL_TEXT | LIST id=kix.qdsepfkil2tz level=0]
Gains: seq13, seq14, seq23, seq22, seq5.

[P01707 | 4960:5003 | NORMAL_TEXT | LIST id=kix.qdsepfkil2tz level=0]
Largest losses: seq6, seq11, seq12, seq21.

[P01708 | 5003:5058 | NORMAL_TEXT | LIST id=kix.qdsepfkil2tz level=0]
Declining sequences: 6, 8, 11, 12, 15, 17, 18, 20, 21.

[P01709 | 5058:5140 | NORMAL_TEXT | LIST id=kix.qdsepfkil2tz level=0]
Paired sequence-Dice change: −0.0137; 95% bootstrap interval: [-0.0667, +0.0355].

[P01710 | 5140:5153 | HEADING_2]
Failure mode

[P01711 | 5156:5169 | NORMAL_TEXT | TABLE row=0 col=0]
Failure mode

[P01712 | 5170:5179 | NORMAL_TEXT | TABLE row=0 col=1]
Evidence

[P01713 | 5180:5190 | NORMAL_TEXT | TABLE row=0 col=2]
Main idea

[P01714 | 5192:5213 | NORMAL_TEXT | TABLE row=1 col=0]
Incomplete prompting

[P01715 | 5214:5303 | NORMAL_TEXT | TABLE row=1 col=1]
Seq15 data-box frame-wise Dice: 0.9664 on one-box frames versus 0.1705 on two-box frames

[P01716 | 5304:5355 | NORMAL_TEXT | TABLE row=1 col=2]
A single box misses additional foreground regions.

[P01717 | 5357:5380 | NORMAL_TEXT | TABLE row=2 col=0]
YOLO empty predictions

[P01718 | 5381:5449 | NORMAL_TEXT | TABLE row=2 col=1]
Seq3: 13/15; seq4: 37/46; seq9: 30/42; seq18: 38/56 positive frames

[P01719 | 5450:5575 | NORMAL_TEXT | TABLE row=2 col=2]
The failure may originate in detection, segmentation, or their hand-off; detector outputs should be logged to separate them.

[P01720 | 5577:5601 | NORMAL_TEXT | TABLE row=3 col=0]
False-positive tracking

[P01721 | 5602:5687 | NORMAL_TEXT | TABLE row=3 col=1]
Seq11 predicts foreground on 90/92 empty-GT frames under both FIFO and ranked memory

[P01722 | 5688:5757 | NORMAL_TEXT | TABLE row=3 col=2]
Temporal propagation persists after the annotated object disappears.

[P01723 | 5759:5795 | NORMAL_TEXT | TABLE row=4 col=0]
Excessive/wrong-region segmentation

[P01724 | 5796:5834 | NORMAL_TEXT | TABLE row=4 col=1]
Seq11 ranked-memory precision: 0.1960

[P01725 | 5835:5899 | NORMAL_TEXT | TABLE row=4 col=2]
Ranked spatial memory frequently predicts incorrect foreground.

[P01726 | 5901:5930 | NORMAL_TEXT | TABLE row=5 col=0]
Inconsistent ranking benefit

[P01727 | 5931:5996 | NORMAL_TEXT | TABLE row=5 col=1]
Positive-frame Dice: seq13 0.0533 → 0.5657; seq6 0.8462 → 0.2732

[P01728 | 5997:6101 | NORMAL_TEXT | TABLE row=5 col=2]
Predicted IoU sometimes recovers tracking, but can also select stale or confidently incorrect memories.

[P01729 | 6102:6103 | NORMAL_TEXT]
⟦EMPTY PARAGRAPH⟧

