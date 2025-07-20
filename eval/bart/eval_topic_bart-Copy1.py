from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="/root/autodl-tmp/model/Facebook/bart-large-mnli")


sequence_to_classify = "Emphasised are features that Microsoft is promising will dramatically speed up the use of its Windows operating system. This update introduces FastTrack mode, which has been previously referred to as CMA mode. At the heart of FastTrack mode is a new power management cap structure where hardware power can be limited by certain high thresholds Fast Track mode will offer several different settings, including a dedicated CPU for higher performance workloads, an unlimited number of Gigabyte Z97 Z97 Extreme 3.5 Quad-Channel DDR3 2667 Memory and dual 8X SATA6x4Gb-channel ports with 2x 6Gb/s USB 3.0 3.0 ports available for fast charging. Hardware features included include an idle feature in which the system will run at idle for some time before restarting the system, integrated Gigabyte's Krait CPU core and an included thermal management feature. There's also an additional Gigabyte Hyper DIMM slot with dual ECC memory modules and a PCI Express 2.0 interconnect adapter, along with Windows 10 Home, 32 Gigabyte memory-copper power card and Gigabyte-designed Gigabyte Precision xo power adapters. FastTrack allows much faster system startup thanks to additional active thread performance settings."

candidate_labels = ['World', 'Sports', 'Business', 'Science', 'Technology']
results = classifier(sequence_to_classify, candidate_labels, multi_label=True, device=0)
print(results)

print(f"Labels: {res['labels']}")
print(f"Scores: {res['scores']}\n")
{'labels': ['travel', 'exploration', 'dancing', 'cooking'],
 'scores': [0.9945111274719238,
  0.9383890628814697,
  0.0057061901316046715,
  0.0018193122232332826],
 'sequence': 'one day I will see the world'}
