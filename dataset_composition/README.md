# New dataset composition analysis 

* Average sequence length of 302.75AA for the training set, 307.4AA for the held out 10k clusters

```bash
python calc_prot_composition.py prostT5_filt_aa.fasta
python calc_prot_composition.py prostT5_filt_ss.fasta
```

* Training set

AA

```bash
A: 0.0846 C: 0.0147 D: 0.0558 E: 0.0617 F: 0.0395 G: 0.0689 H: 0.0222 I: 0.0547 K: 0.0515 L: 0.0947 M: 0.0217 N: 0.0419 P: 0.0507 Q: 0.0381 R: 0.0585 S: 0.0738 T: 0.0575 V: 0.0652 W: 0.0137 Y: 0.0306 
```

3Di

```bash
A: 0.0283 C: 0.0290 D: 0.2436 E: 0.0128 F: 0.0189 G: 0.0219 H: 0.0228 I: 0.0191 K: 0.0168 L: 0.0630 M: 0.0068 N: 0.0229 P: 0.1059 Q: 0.0404 R: 0.0248 S: 0.0583 T: 0.0157 V: 0.2156 W: 0.0194 Y: 0.0140 
```

* More D's (loopy) at the expense of V's (helices) compared to ProstT5 and PDB (compared to the ProstT5 Supplementary Figure)


## singletons

* 21.8M of them
* Average length of 162.79 AA
* Even more extreme upweighting of D's and V's
* Note these were not used for training in the end - but can still be found on HuggingFace if super keen

AA

```bash
A: 0.0838 C: 0.0165 D: 0.0530 E: 0.0613 F: 0.0380 G: 0.0675 H: 0.0233 I: 0.0514 K: 0.0522 L: 0.0904 M: 0.0233 N: 0.0413 P: 0.0547 Q: 0.0399 R: 0.0634 S: 0.0790 T: 0.0579 V: 0.0618 W: 0.0133 Y: 0.0279 
```

3Di

```bash
A: 0.0159 C: 0.0232 D: 0.3190 E: 0.0067 F: 0.0122 G: 0.0142 H: 0.0128 I: 0.0118 K: 0.0095 L: 0.0326 M: 0.0036 N: 0.0135 P: 0.1443 Q: 0.0236 R: 0.0149 S: 0.0298 T: 0.0115 V: 0.2796 W: 0.0139 Y: 0.0075 
```
