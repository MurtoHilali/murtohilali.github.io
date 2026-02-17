---
layout: post
title: How I found 71 natural alternatives to a $200 million drug
description: A step-by-step drug discovery and AI virtual screening project done from my laptop
tags: bioinformatics "drug discovery" "virtual screening" project
image: /img/seo/aadd.png
thumb: /img/thumb/aadd.webp
expanded: /img/expanded/aadd-expanded.png
---
[![](https://substackcdn.com/image/fetch/$s_!Yu60!,w_2400,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9ac0f9d4-1c03-4370-87c7-f888ef6e0474_960x540.png)](https://substackcdn.com/image/fetch/$s_!Yu60!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9ac0f9d4-1c03-4370-87c7-f888ef6e0474_960x540.png)
By Author — Not random images either, there’s a chance you find the compounds in all of these organisms.

[![Static Badge](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/MurtoHilali/Acarbose-Alternative-Drug-Discovery/tree/main)

## Table of Contents

1. **[Exploratory Data Analysis for COCONUT](#1-exploratory-data-analysis-for-coconut)** — Profiling ~100K natural products to understand drug-likeness, physicochemical space, and screening viability.
2. **[Chemical Similarity Search](#2-chemical-similarity-search)** — Fingerprint-based filtering (ECFP, MACCS, pharmacophore, USRCAT) to reduce the database to high-similarity candidates.
3. **[Protein Model Selection, Cleanup, and Preprocessing](#3-protein-model-selection-cleanup-and-preprocessing)** — Selecting acarbose-bound crystal structures and preparing them for docking.
3. **[Protein–Ligand Docking & Assessment](#4-protein-ligand-docking--assessment)** — Running GNINA-based virtual screening and scoring with CNN-VS.
4. **[Lead Validation and Controls](#5-lead-validation-and-controls)** — Decoys, redocking replicates, statistical filtering, and stability analysis to remove false positives.
5. **[Candidate Overview & Key Takeaways](#candidate-overview--key-takeaways)** — Final shortlist by target protein and reflections on the workflow.

Flavonoids are a large, diverse group of secondary plant metabolites that play a role in pigmentation, UV protection, and insecticidal activity. Specifically, **some anthocyanins (a subclass) can** **inhibit insect digestive enzymes**, preventing nutrient uptake and development, serving as a defence mechanism against predators.

Obviously, there’s potential here for bug Ozempic (or insecticides, you pick). But research on _Allium mongolicum_ flowers has also revealed that their **flavonoids may inhibit** _**human**_ **digestive enzymes**, including our very own starch-metabolizing **α-glucosidase**. These compounds could be used to treat diabetes, as inhibiting starch breakdown reduces postprandial blood glucose levels ([Li et al., 2025](https://link.springer.com/article/10.1007/s11130-025-01422-8)).

Initial docking analysis revealed that _A. mongolicum_-derived **isoquercetin** (already in clinical trials) can inhibit α-glucosidase (AGI) activity with an effect similar to that of **acarbose** , a common AGI derived from _Actinoplanes_ bacteria. Chemical similarity search revealed that **troxerutin** (aka vitamin P4, a flavonoid you can buy over the counter) shares some substructural motifs with isoquercetin, _and_ its docking analysis showed similar binding affinity.

[![](https://substackcdn.com/image/fetch/$s_!jfor!,w_2400,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0346ecbb-5683-459b-bd35-5f382e251d54_1327x744.png)](https://substackcdn.com/image/fetch/$s_!jfor!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0346ecbb-5683-459b-bd35-5f382e251d54_1327x744.png)Docking results from DiffDock-Pocket.

>  **Note** : troxerutin’s slightly higher affinity may be attributed to steric interactions introduced by the moieties found on the larger molecule.

 **I wanted to take this approach and extend it to a larger database of natural compounds** to find other biochemicals that could hypothetically outperform acarbose — after all, the drug is projected to [have a $220 million global market by 2035](https://www.businessresearchinsights.com/market-reports/acarbose-market-111352). 

Here’s my overall process:

  1.  **Exploratory data analysis of [COCONUT](https://coconut.naturalproducts.net/)**, a natural products database.

  2.  **Chemical similarity search** of acarbose against COCONUT to find structurally similar NPs.

  3.  **Protein model selection, cleanup, and preprocessing**.

  4.  **Protein-ligand docking and assessment** , where I run a virtual screen against human α-glucosidases.

  5.  **Lead validation and controls** to identify any false positives.




You can see all the code (in progress) [here](https://github.com/MurtoHilali/Acarbose-Alternative-Drug-Discovery/tree/main).

[![](https://substackcdn.com/image/fetch/$s_!FE0O!,w_2400,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3580a2c-51f5-4731-ac22-c51910f44289_1462x530.png)](https://substackcdn.com/image/fetch/$s_!FE0O!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3580a2c-51f5-4731-ac22-c51910f44289_1462x530.png) Workflow diagram

## 1\. Exploratory data analysis for COCONUT

In this section, I’ll take a look at:

  * Distributions of some pre-existing features in the database.

  * The spread of some derived ratios.




There are a little over 100,000 natural products in the COCONUT database. I started by doing some simple feature visualization:

[![](https://substackcdn.com/image/fetch/$s_!vfUB!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcff909f3-176e-45c9-9a20-13e4a82b0a9e_1273x1317.png)](https://substackcdn.com/image/fetch/$s_!vfUB!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcff909f3-176e-45c9-9a20-13e4a82b0a9e_1273x1317.png)

  *  **Total atom count** : all atoms, including hydrogen.

  *  **Heavy atom count** : any non-hydrogen atom.

  *  **Molecular weight** : in grams per mole

  *  **ALogP** : A drug discovery metric that estimates molecular lipophilicity.

  *  **Topological polar surface area** : The surface sum of all polar atoms, also a measure for drug bioavailability.

  *  **Rotatable bond count** : single bonds that allow for free rotation, generally a measure of flexibility/rigidity.

  *  **HBAs/HBDs** : Polarity metrics — we typically want a particular balance of the two, with more HBAs than HBDs in drug-like molecules.

  *  **Lipinski Rule of Five violations** : the RO5 is a rule of thumb for evaluating drug-likeness based on a set of molecular properties. However, it is worth noting that several drugs do not have these characteristics — in other words, it is not the be-all and end-all. Those properties are:
     * 5 or fewer HBDs
     * No more than 10 HBAs
     * Molecular mass under 500 Da
     * ClogP < 5

  *  **Aromatic rings count** : Aromatic rings add planarity to a molecule.

  *  **QED druglikeness** : A drug likeness measurement that focuses on 8 features (all listed attributes other than RO5 and atom counts).

  *  **NP likeness** : Natural product likeness, measures a molecule’s similarity to known natural products.


For a closer look at some derived ratios, I looked at the following:

[![](https://substackcdn.com/image/fetch/$s_!Pute!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F229da3a2-8ad7-44fd-8418-defa3d07b52a_597x455.png)](https://substackcdn.com/image/fetch/$s_!Pute!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F229da3a2-8ad7-44fd-8418-defa3d07b52a_597x455.png)

Research indicates that a TPSA/MW ratio >= 0.2 is ideal for good solubility; approximately half of the dataset meets this criterion ([Whitty et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5821503)).

[![](https://substackcdn.com/image/fetch/$s_!yLCW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1d233e82-e5e8-46a8-b9be-a2971549a7b4_597x455.png)](https://substackcdn.com/image/fetch/$s_!yLCW!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1d233e82-e5e8-46a8-b9be-a2971549a7b4_597x455.png)

This metric stems from two findings:

  * Fsp3 (the fraction of sp3 carbons over total carbons) increases as drugs move through the discovery pipeline ([Lovering et al., 2009](https://pubs.acs.org/doi/10.1021/jm901241e)).

  * Molecules with more than 3 rings tend to have poor solubility, high lipophilicity, and high promiscuity (i.e., off-target binding) ([Ritchie et al., 2011](https://pubmed.ncbi.nlm.nih.gov/21129497/)).




Here, I take the ratio of both in the dataset. Two populations emerge: many NPs with no saturated carbons and a smaller group with some balance.

 **Takeaways** :

  * COCONUT offers a relatively diverse search space
  * There appears to be drug-like potential amongst several NPs


## 2\. Chemical similarity search

Chemical similarity is generally determined by generating chemical fingerprints for a molecule, **fixed-size identifiers that capture information about atomic structure and chemistry**. Several fingerprinting protocols exist, each with its own strengths and weaknesses. Here are the ones I chose and why:

  *  **ECFP**. Extended-Connectivity Fingerprints represent molecular structures and are widely used in cheminformatics. They enable substructure and similarity searching but are also used for quantitative structure-activity relationship (QSAR) modelling. If you’re interested in molecular activity, these are a good pick.

  *  **Pharmacophoric features**. These fingerprints encode data on chemical features typically involved in pharmacological actions, such as hydrophobicity, charge, aromaticity, etc. This makes them useful when we’re looking for chemicals with properties similar to those of acarbose.

  *  **MACCS keys**. Short for Molecular Access System, MACCS are one of the simpler fingerprinting methods. They encode the answers to established T/F questions about chemical structures, such as the ring size, ion presence, or oxygen count. They can be generated quite quickly and are relatively interpretable.

  *  **Ultrafast Shape Recognition**. USR is the only 3D method on this list. It generates a vector of shape descriptors for a set of conformers of a given molecule. USRCAT, which I am using more specifically, also encodes pharmacological features.




By generating fingerprints for acarbose, we can use **Tanimoto similarity to assess the degree of agreement** between our database compounds and the lead target. USR is the exception; we use inverse Manhattan distance instead.

To trim the dataset down, I used the following cutoffs for Tanimoto similarity:

  *  **ECFP: 0.4**. Research seems to show enrichments of similarly active compounds beyond a TC of 0.4 with [minimal false positives](https://pmc.ncbi.nlm.nih.gov/articles/PMC12370643/#:~:text=FIGURE%207.&text=\(A\)%20Precision%E2%80%93recall%20curves,TargetHunter%2C%20MolTarPred%2C%20and%20TarPred.).

  *  **MACCS: 0.7**. See above.

  *  **Pharmacophoric features: 0.8**. A slightly more arbitrary cutoff, but I elected to be stricter since ECFP and MACCS are fairly generous.

  *  **USRCAT: top 1%**. In virtual screening workflows and decoy comparison benchmarks, the top 1% of candidates are [selected](https://pmc.ncbi.nlm.nih.gov/articles/PMC3505738/).




This gave me a dataset of 10,341 high-similarity candidate molecules to explore. Ligands were generated using RDKit and converted to `.pdbqt `using OpenBabel.

## 3\. Protein model selection, cleanup, and preprocessing

Acarbose is a bit of a rolling stone, it can’t be tied down — by which I mean it **interacts with multiple proteins**. To be thorough, I wanted to explore this virtual screening process across all of its protein targets at once. In hindsight, this made it much harder to keep track of (I’ll take it one target at a time from now on).

I made my selections based on protein targets listed in [DrugBank](https://go.drugbank.com/drugs/DB00284) and the resolution of the corresponding protein model. **I also specifically selected PDB entries of the target protein** _ **in complex with acarbose**_ so I’d a) have a defined binding pocket and b) have an experimental reference point for future validations. Finally, all proteins were put through [PDB-redo](https://pdb-redo.eu/) before being converted to `.pdbqt `for docking:

  1.  **[2QMJ](https://www.rcsb.org/structure/2QMJ)** : The N-terminal subunit of human maltase-glucoamylase in complex with acarbose. As you can see from the PDB page, there’s room for growth in model quality, but PDB-redo led to some minor improvements. (Gene: MGAM).
  2.  **[3BAJ](https://www.rcsb.org/structure/3BAJ)** : Human pancreatic alpha-amylase in complex with nitrate and acarbose. PDB-redo led to more significant improvements here. (Gene: AMY2A).
  3.  **[5NN8](https://www.rcsb.org/structure/5NN8)** : Human lysosomal acid-alpha-glucosidase in complex with acarbose (Gene: GAA).


Below, **acarbose is designated in light green** with binding residues (within 4 Å of acarbose) in red. All other observed elements are ions and ligands, which are removed for docking.

[![](https://substackcdn.com/image/fetch/$s_!gvGx!,w_2400,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdfda8e44-577e-4df3-8306-ac86226edf20_1285x516.png)](https://substackcdn.com/image/fetch/$s_!gvGx!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdfda8e44-577e-4df3-8306-ac86226edf20_1285x516.png)
Note: one of the highlighted light green ligands in 5NN8 is an acarbose-derived trisaccharide; the other binding pocket (containing true alpha-acarbose) is used for docking configuration — Images generated via [The Protein Imager](https://academic.oup.com/bioinformatics/article/36/9/2909/5701652). 


## 4\. Protein-ligand docking & assessment

[![](https://substackcdn.com/image/fetch/$s_!lsW5!,w_2400,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F91dd45d6-a4e7-4f07-bf70-cdb119fea0d9_1305x545.png)](https://substackcdn.com/image/fetch/$s_!lsW5!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F91dd45d6-a4e7-4f07-bf70-cdb119fea0d9_1305x545.png)Figures generated using ChimeraX. Can you tell I tried (unsuccessfully) to create Goodsell-like graphics?

 **For protein-ligand docking, I used[GNINA](https://github.com/gnina/gnina)**, an open-source AutoDock Vina fork that uses CNNs to score ligand poses.

Since we know where acarbose is supposed to bind (the protein models include acarbose), we can **guide the docking process by setting an autobox** (basically a mini 3D search space) around its coordinates. In the diagrams above, I’ve indicated the autobox GNINA draws around the ligands (4 Å outward from the farthest corners of the molecule) in red.

While it may be the most computationally expensive step, this stage of the project was actually the easiest. I ran GNINA on an HPC cluster in parallel fashion to save time — although GNINA wasn’t designed specifically for high-throughput virtual screening[1](https://offbase.substack.com/p/how-i-found-71-natural-alternatives#footnote-1-187562974), this step was completed in approximately one day.

## 5\. Lead validation and controls

In this section, I go over:

  * Filtering through the results of our initial docking run.

  * Analyzing the results of decoy docking tests.

  * Compare lead ligand performance across replicates.




 **GNINA generates several metrics for binding poses** , including CNN pose score (the AI’s assessment for score quality) and CNN binding affinity (the AI-predicted kcal/mol). Benchmarking studies with GNINA have indicated that the product of these two values, **CNN_VS (virtual screening), can be used to identify potentially active ligands** [when it exceeds 6.30](https://www.mdpi.com/1420-3049/30/16/3361).

[![](https://substackcdn.com/image/fetch/$s_!49n-!,w_2400,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F66d77ae7-1def-42af-8daa-84a9b53ebc74_1489x390.png)](https://substackcdn.com/image/fetch/$s_!49n-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F66d77ae7-1def-42af-8daa-84a9b53ebc74_1489x390.png)Horizontal lines represent the CNN-VS threshold of 6.30.

Taking the CNN_VS of all ligands docked against each protein, **71 ligands passed the threshold.**

Okay, the headline promise-payoff is complete—but we’re not done yet. After collecting a set of leads, I needed to validate some of the findings. There were a few ways I went about doing this, with each control telling me something different:

  *  **Molecular decoys**. For each lead, I used [LUDe](https://www.sciencedirect.com/science/article/pii/S2667318525000054) to generate physicochemically similar molecules for docking against the protein receptors. Decoy molecules tell us whether there’s something unique about a lead or if any molecule with a similar molecular weight, log P, polarity, etc., would perform just as well.

  *  **Redocking with variant seeds**. By using different seeds, we can check whether the results from our first run are consistent or a fluke — in other words, we need to run replicates.

  *  **Structural interaction fingerprint comparison**. The PLECFP, or protein-ligand extended connectivity fingerprint, is a method for hashing protein-ligand interactions for comparison. Here, I used it to compare structural interactions between the leads and acarbose during protein binding.




Here are the results for each:

### Molecular decoys

 **Decoys help us determine whether our docking results are false positives**. By matching our lead molecules against decoy molecules with similar properties (e.g., molecular weight, number of rotatable bonds, log P), we can assess whether our leads have real potential.

I used LUDe (LIDEB’s Useful Decoys) to generate decoys for all (putatively) biologically active ligands. **LUDe provides fifty decoys per ligand, all of which were docked against the protein models using GNINA** under the same parameters, with CNN-VS taken as well.

[![](https://substackcdn.com/image/fetch/$s_!TUfE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F63e2d31a-da5a-4358-b0e7-280800ba18b6_1489x989.png)](https://substackcdn.com/image/fetch/$s_!TUfE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F63e2d31a-da5a-4358-b0e7-280800ba18b6_1489x989.png)Comparative overall performance (measured via CNN-VS) of leads vs. property-matched decoys.

Alternatively:

[![](https://substackcdn.com/image/fetch/$s_!Tno9!,w_2400,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0d6b83a-b39e-4988-be36-d32c5457414a_1489x490.png)](https://substackcdn.com/image/fetch/$s_!Tno9!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb0d6b83a-b39e-4988-be36-d32c5457414a_1489x490.png)

As you can see, nearly all decoy poses scored below our activity threshold. However, to be more robust, I did the following checks:

  * Get a one-sided empirical p-value, robust z-score, and Benjamini-Hochberg adjusted q-value for each ligand vs. decoy set.

  * Determine the top-1 enrichment for each ligand vs. decoy set (or, how often does the ligand outrank its decoys in CNN-VS).




#### P-values & z-scores

 **For each ligand, I compared CNN-VS scores with ~50 matched decoys** (in some cases, RDKit was unable to generate conformers for the decoys) and computed a one-sided empirical p-value. I also determined how many robust standard deviations the ligand lay above the median and adjusted all p-values using the Benjamini-Hochberg false discovery rate.

  *  **2QMJ & 5NN8**: For all ligands, all adjusted p-values < 0.05.

  *  **3BAJ** : Out of 53 ligands, one p-value > 0.05.




Some results for 3BAJ are shown below.

[![](https://substackcdn.com/image/fetch/$s_!6H0T!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F667a7b3a-99fd-4b5f-bf17-b3533d9b333d_502x854.png)](https://substackcdn.com/image/fetch/$s_!6H0T!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F667a7b3a-99fd-4b5f-bf17-b3533d9b333d_502x854.png)

#### Ligand dominance, or rank-based enrichment

Realistically, **the ligand vs. decoy sets are too small for any real statistical robustness.** A more useful heuristic is to compare the ligands’ ranks relative to their matched decoys when ordered by CNN-VS.

 **For all three proteins, all leads ranked at or above the 98th percentile relative to their decoys.** Actually, all except 2 leads for 3BAJ ranked first relative to their decoys. Some results for 3BAJ are shown below:

[![](https://substackcdn.com/image/fetch/$s_!3gjx!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe3045ea2-f92c-4b4a-9d0e-379897635981_1166x676.png)](https://substackcdn.com/image/fetch/$s_!3gjx!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe3045ea2-f92c-4b4a-9d0e-379897635981_1166x676.png)You’ll notice very few decoys were generated for CNP0064234.2; it was not considered a candidate for 3BAJ.

### Redocking with variant seeds

One of the more important validation tests, **this process is meant to tell us if the leads we determined were flukes or are consistently high-CNN-VS across replicates** , an attribute we’ll refer to as stability from here on.

 **Once again, this control helps us filter out false positives.** If expanded, it can help us identify false negatives: ligands that didn’t pass the CNN-VS threshold in the original screen but would otherwise. I chose not to do that in this project due to time and computational cost constraints, but it would be a logical next step.

For this control, I simply **took all of our leads and docked them again with GNINA using 5 variant seeds**. Below are the distributions of CNN-VS scores for the initial run compared with the redocked runs:

[![](https://substackcdn.com/image/fetch/$s_!yToO!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0bb7a111-c296-498a-9788-bd0096fca1a9_1600x526.png)](https://substackcdn.com/image/fetch/$s_!yToO!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0bb7a111-c296-498a-9788-bd0096fca1a9_1600x526.png)

As we can see, the CNN-VS for the redocked ligands exhibits a wider distribution (as expected — we haven’t artificially subset them to CNN-VS >= 6.30, and there are 25 times as many data points).

Let’s take a closer look:

[![](https://substackcdn.com/image/fetch/$s_!MnWI!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd2c331f7-cc5f-4cc4-97a9-87ac0b135ece_1587x1389.png)](https://substackcdn.com/image/fetch/$s_!MnWI!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd2c331f7-cc5f-4cc4-97a9-87ac0b135ece_1587x1389.png) _5.1. Jittered Box-Plot of CNN-VS Distribution by Identifier Across All Seeds_

 **What this shows:**

Each box shows the spread of CNN-VS scores for a given lead identifier across all seeds, including the original and redocking runs.

  * Identifiers are ranked by median CNN-VS.

  * This gives us a sense of consistency vs variance per ligand.




Already, we can see several leads do not pass muster.

 **Why it matters:**

These charts help us see which identifiers have the most consistent CNN-VS scores and whether their distributions exceed the biologically active threshold.

[![](https://substackcdn.com/image/fetch/$s_!LFvZ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F70a7e351-d150-4d01-96b3-3a873438d7cf_790x1590.png)](https://substackcdn.com/image/fetch/$s_!LFvZ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F70a7e351-d150-4d01-96b3-3a873438d7cf_790x1590.png) _5.2. Spaghetti Plot of Best CNN-VS Score by Identifier Across All Seeds_

 **What this shows** :

This is a closer look than the previous boxplot and shows us seeds on the x-axis. Once again, it shows us the consistency and potency of each identifier:

  * Which ligands are seed-stable vs. seed-sensitive?

  * How often are trajectories exceeding 6.30?

  * Are high medians due to lucky seeds or consistency?




>  **Note** : It might make more sense to use a spaghetti plot for fewer samples that were actually labelled, but here it’s meant to provide a broader overview.

 **Why it matters** :

These plots show whether CNN-VS scores are consistent across runs and how high or low those scores are. We’re looking for consistent scoring that’s also above the threshold.

[![](https://substackcdn.com/image/fetch/$s_!b0My!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb05d105a-2e3c-4f16-a2c0-d6572562b7b2_1600x476.png)](https://substackcdn.com/image/fetch/$s_!b0My!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb05d105a-2e3c-4f16-a2c0-d6572562b7b2_1600x476.png) _5.3. Scatter Plot of Cross-Seed Median CNN-VS Score & Cross-Seed Interquartile Range_

 **What this shows** :

The median CNN-VS across seeds provides a general idea of the CNN-VS for a particular ligand; the IQR indicates the stability of that CNN-VS. A lower IQR means a tighter distribution.

  * Bottom-right quadrant: low variance, high CNN-VS (most promising candidates).

  * Top-right quadrant: high variance, high CNN-VS (false positives).

  * Top-left quadrant: high variance, low CNN-VS (discard).

  * Bottom-left quadrant: low variance, low CNN-VS (true negatives)

  * Are high medians due to lucky seeds or consistency?




 **Why it matters** :

This is another way of visualizing consistency (low IQR) and quality (median CNN-VS).

[![](https://substackcdn.com/image/fetch/$s_!y572!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F150e423e-2b1b-4b2a-b0a0-8864bedacc85_1589x1389.png)](https://substackcdn.com/image/fetch/$s_!y572!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F150e423e-2b1b-4b2a-b0a0-8864bedacc85_1589x1389.png) _5.4. Median Comparison Between Originals and Redock CNN-VS by Identifier_

 **What this shows** :

One of the simpler visualizations; all we see here is the difference between the original run’s CNN-VS and the median redock runs’ CNN-VS.

  * A small drop tells us our original run for that ligand was reliable.

  * A big drop means it got lucky.




This plot becomes more useful if we expand our initial redocking set to include those leads with CNN-VS scores below 6.30 — a jump would indicate a false negative.

 **Why it matters** :

A companion to IQR vs. CNN-VS. A snapshot of pose stability.

Taking all of this into account, my final filtration process was to subset to the leads that scored CNN-VS > 6.30 in 5 out of 6 replicates, then rank them by the median CNN-VS-to-interquartile range ratio. That meant the top 3 candidates for each protein were as follows:

 **2QMJ** :

  * CNP0111740.3

  * CNP0475805.2 (Phaeospelide A)

  * CNP0233949.3 (Convallataxol)




 **3BAJ** :

  * CNP0242138.5

  * CNP0146238.1

  * CNP0185842.19




 **5NN8** :

  * CNP0533794.1

  * CNP0234543.1 (2-deoxyecdysterone 20,20-monoacetonide)

  * CNP0111740.3




We’ll take a closer look at these soon — but first, let’s do one final validation.

### Structural interaction fingerprint comparison

The goal of this step is more exploratory than confirmatory. My aim here was to **determine whether the lead ligands interact with the target proteins similarly to experimental acarbose.** We don’t exactly need to prioritize the ligands with similar binding profiles, as it could be expected that a more performant ligand would behave differently.

We can compare ligand-protein structural interactions across a diverse set of ligand sizes using **structural interaction fingerprints**. These go one step beyond typical molecular fingerprints and **encode molecular interactions** between ligands and binding-pocket residues, hashing them to preset sizes to enable comparisons.

 **It seems the best fingerprint so far is[PLECFP](https://pmc.ncbi.nlm.nih.gov/articles/PMC10813698/)**, which combines the best features of all previous methods — atomic environment capture, expanded capture radius, etc. It’s shown promise in AI tasks, lead optimization, and scaffold hopping. That’s why I chose to use it for my initial pass:

[![](https://substackcdn.com/image/fetch/$s_!TAXY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F95194d8d-6fba-4fb8-9367-3b61e5a78644_1490x390.png)](https://substackcdn.com/image/fetch/$s_!TAXY!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F95194d8d-6fba-4fb8-9367-3b61e5a78644_1490x390.png)

As you can see, **our similarity scores are practically zero**. (Full disclosure: this was a pretty disappointing chart to generate). 

These results are a little surprising — the initial list of 10,000 candidates was selected because they shared similarities with acarbose, so we’d expect at least _some_ overlap.

Moreover, when creating a residue contact map for each ligand against its target protein, the interacting residues seem quite similar:

 **2QMJ** :

[![](https://substackcdn.com/image/fetch/$s_!QTYj!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7fe8e9c7-3a15-42c2-bc57-2fe2ceb3a9a4_1600x432.png)](https://substackcdn.com/image/fetch/$s_!QTYj!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7fe8e9c7-3a15-42c2-bc57-2fe2ceb3a9a4_1600x432.png)

 **3BAJ** :

[![](https://substackcdn.com/image/fetch/$s_!1aPG!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9e3871ae-6c9d-4b6d-831f-68f97c646965_1600x1482.png)](https://substackcdn.com/image/fetch/$s_!1aPG!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9e3871ae-6c9d-4b6d-831f-68f97c646965_1600x1482.png)

 **5NN8** :

[![](https://substackcdn.com/image/fetch/$s_!2SVS!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe0c76b2c-6981-4184-98b5-ff55910f96e6_1600x430.png)](https://substackcdn.com/image/fetch/$s_!2SVS!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe0c76b2c-6981-4184-98b5-ff55910f96e6_1600x430.png)

I’m still not sure why this might be, so please let me know if you have an explanation!

Also, out of curiosity, I determined pairwise Tanimoto similarity across all leads[2](https://offbase.substack.com/p/how-i-found-71-natural-alternatives#footnote-2-187562974). There seems to be clustering among some 3BAJ leads, which may merit further investigation.

Some of the lead ligands for 3BAJ, such as CNP0146238.29, appear to share binding behaviours with other leads, such as CNP0156494 — this may be something worth further investigation.

## Candidate overview & key takeaways

Let’s start by interpreting our shortlist from earlier:

### 2QMJ

  *  **CNP0111740.3** : An oleanane triterpenoid that shares parentage with Ilekudinoside I, and seems to appear in traditional Chinese medicine.

  *  **CNP0475805.2** (Phaeospelide A): A polyene macrolide from _Arthrinium phaeospermum_ , a hairy-caterpillar-associated fungus. Although the genes required to express this molecule appear silent in the fungus, they were [heterologously expressed in bacteria](https://pubs.acs.org/doi/10.1021/acs.orglett.9b01674). This molecule also exhibits binding affinity with 5NN8.

  *  **CNP0233949.3** (Convallataxol): The best-annotated NP on this list, convallataxol, is a steroid glycoside [found in the leaves and flowers](https://www.sciencedirect.com/science/article/abs/pii/S0968089614000893) of _Antiaris toxicaria_ , a highly poisonous tree.


### 3BAJ

  *  **CNP0242138.5** : Another cardenolide from the _Nerium indicum_ and _Nerium oleander_ species, flowering shrubs used in traditional medicine.

  *  **CNP0146238.1** : An ecdysteroid that appears to be used in Ethiopian traditional medicine.

  *  **CNP0185842.19** : A cardenolide that seems to be a stereo variant of beauwalloside, which is also found in _N. oleander_.


### 5NN8

  *  **CNP0533794.1:** An open-chain polyketide — no other documentation found.

  *  **CNP0234543.1** (2-deoxyecdysterone 20,20-monoacetonide): An ecdysteroid from the roots of the _Silene brahuica_ plant, shown to have some [anti-inflammatory properties](https://link.springer.com/article/10.1007/BF00629760).

  *  **CNP0111740.3** (See earlier)


Some honourable mentions:

  *  **CNP0156494.1** : An ecdysteroid from _Vitex polygama_ , which appears to be [shidasterone](https://www.degruyterbrill.com/document/doi/10.1515/znc-2008-5-611/html), which seems to have enzyme-inhibitory effects.

  *  **CNP0279421.3** : Torvoside K, a [saponin from many species](https://www.sciencedirect.com/science/article/abs/pii/S0031942211000574), including ones of the _Solanum_ genus, which seems to have antifungal properties.


If you’re in a lab, made it this far, and have access to such delightful flora as _A. toxicaria_ or a library of traditional Ethiopian medicine, feel free to take this list of lead candidates and get to testing.

Some of my key takeaways:

  *  **The easiest part of virtual screening is the docking**. The real work involves preparation, preprocessing, post-docking controls, and validation.

  *  **Many natural products lack sufficient research, even though they present countless therapeutic opportunities**. Admittedly, this is largely due to extraction and isolation challenges — but there’s an opportunity here nonetheless.

  *  **Good research involves lots of small, smart decisions**. Several times during this project (not documented here), I had to back up and redo an entire step after rushing through a small decision. Better planning and research ahead of time would have saved time and boosted my efficiency.




Thank you for reading! Am I a genius? A modern Prometheus? Have I made grievous and unforgivable scientific errors? Please let me know if you have any feedback.

[1](https://offbase.substack.com/p/how-i-found-71-natural-alternatives#footnote-anchor-1-187562974)

They actually [specifically recommend against](https://gnina.github.io/gnina/rsc_workshop2021/#/69) using GNINA for high-throughput screening applications. However, simple parallelization allows you to spawn multiple GNINA processes on a single GPU node, so you can take advantage of the program's speed.

[2](https://offbase.substack.com/p/how-i-found-71-natural-alternatives#footnote-anchor-2-187562974)

 **2BAJ** :

[![](https://substackcdn.com/image/fetch/$s_!qiup!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3b57949a-d54f-40d4-b068-acb5f5c5a80c_766x689.png)](https://substackcdn.com/image/fetch/$s_!qiup!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3b57949a-d54f-40d4-b068-acb5f5c5a80c_766x689.png)

 **3BAJ** :

[![](https://substackcdn.com/image/fetch/$s_!IF_J!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d97edfd-e658-43cd-8292-bd4bedaf3303_767x690.png)](https://substackcdn.com/image/fetch/$s_!IF_J!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d97edfd-e658-43cd-8292-bd4bedaf3303_767x690.png)

 **5NN8:**

[![](https://substackcdn.com/image/fetch/$s_!McXW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F638fdd51-5d66-49e1-a4a7-40964894fe04_767x690.png)](https://substackcdn.com/image/fetch/$s_!McXW!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F638fdd51-5d66-49e1-a4a7-40964894fe04_767x690.png)
