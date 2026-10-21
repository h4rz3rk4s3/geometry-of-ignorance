We thank the reviewers for their thorough and constructive feedback. There are many great suggestions for improvement that we can easily integrate in a major revision, given the additional page. These represent important clarifications, but they don't threaten or invalidate our findings.

#### RA-Q1/W1/W4
We are happy to clarify: The 22 uncertainty cues had very little impact and don't change the interpretation of RQ3. We aggregated SHAP attributions over the full ground-truth and out-of-distribution datasets to identify the individual tokens with the strongest contribution towards a non-knowledge prediction. Only 4 of the 15 strongest tokens in the ground-truth dataset were search cues (weird, suddenly, seems, how), and only 2 in the out-of-distribution dataset (maybe, seems). All others are uncertainty or surprise markers outside the search vocabulary. SHAP also reveals that none of these 6 cue tokens had strong attribution values. Instead, attribution is spread over multi-token spans (e.g., "I don't remember what" in Fig. 5) that jointly indicate non-knowledge. Thus, the models didn't learn the search-cue list, but local n-gram patterns. We will add these results to Sec. IV-C. This doesn’t change the interpretation of RQ3.

#### RA-Q2/RB-W3
Good point: The 101 ground-truth examples and the 101 expert-labeled examples are the same. Hence, the ground-truth evaluation uses all expert-labeled examples. The ground-truth examples were used as seeds to generate the synthetic examples, but no ground-truth example is itself part of the synthetic dataset. Hence, there is no direct leakage, but potential indirect leakage if synthetic examples contain longer passages from the ground-truth example they were seeded from. We mitigated this risk by applying ROUGE-L filtering. The impact on the ground-truth evaluation is therefore reduced. We will clarify this in the threats to validity.

#### RA-Q3/W2
We thank the reviewer for suggesting bootstrapped CIs. We computed class-stratified BCa bootstrap 95% CIs [1] to account for the 35/14 class imbalance in the out-of-distribution dataset. The F1-scores with CIs are: ModernBERT-NK 0.60 [0.47, 0.73], RoBERTa-NK-Base 0.73 [0.60, 0.83], and RoBERTa-NK-Large 0.81 [0.68, 0.90]. We will add these results to our paper, but they don't change our interpretation of the results.

#### RA-W3
We thank the reviewer for pointing out that we didn't explain this. We will clarify that we use LMs due to their much smaller size, which allows us to analyze large-scale datasets efficiently with modest compute, which we plan to do in future work. Small LLMs aren't a feasible alternative, as they tend to produce ill-formatted outputs that are expensive to handle.

#### RB-Q1/W1
Prior SE work on confusion and developer questions treats knowledge gaps as individual, problematic states. Non-knowledge extends this perspective: for example, it relates individual and community knowledge, captures that such states can be actively resolved or deliberately maintained, and represents knowledge gaps that developers aren't even aware of. This adds substantial new perspectives to SE research, for example: Is confusion caused by a community actively maintaining non-knowledge? When and how do developers become aware of their non-knowledge? How do different manifestations of non-knowledge impact how software is built? We will clarify this in the paper.

#### RB-Q2/W2
From each of the four repositories, we collected the 100 threads with the most comments. One annotator opened each thread and excluded those consisting mostly of bot messages without developer discussion. Additionally, we searched for threads that contained any of the 22 target words. Unfortunately, we didn't document how many threads each step excluded. In total, 150 threads remained. One annotator inspected these and excluded 103 as not fitting the non-knowledge definitions. The remaining 47 threads were jointly analyzed by the two annotators through collaborative open coding. We extracted the 101 ground-truth examples from these 47 threads. Unfortunately, as this wasn't part of the open coding, we didn't document how many messages we discarded or considered ambiguous. We will add these details to Sec. III-A/E.

#### RB-Q3/W3
We acknowledge that we have not been fully clear: Our data split assigned 83 ground-truth examples to training (mean 121.2 synthetic examples each, SD 46.1, range 61–199, pooled over both generators, including neutral examples and 2:1 oversampling) and 18 to the synthetic test split (mean 177, SD 15.6, range 143–197). Thus, 83 ground-truth examples have synthetic examples in the training data. Therefore, the ground-truth F1 (0.96) only shows whether performance survives the transfer from synthetic to real data. We make this point in the paper, but agree that it is currently not described clearly enough.

#### RB-W4
Existing research on the linguistics of non-knowledge shows that non-knowledge has a prototypical vocabulary but is mainly expressed through morpho-syntactic patterns (e.g., "that must be why") and rhetorical figures [2], and that uncertainty cues don't transfer well across domains [3]. We therefore expect a lexical matcher to produce many false positives, as it cannot evaluate the context needed to detect non-knowledge consistently.

#### RB-Q4
We apologize that we forgot to add a README file to our replication package, and we are sorry if this made your work more difficult. The Zenodo version will include a README with file-level pointers to make it easier for other researchers to use and replicate our work.

#### RC-W1
We fully agree that this is important work, yet it requires a reliable way to detect non-knowledge in SE at scale. Our paper contributes the prerequisites for such work. Please consider that without first demonstrating that our approach works, we cannot confidently pursue the follow-up work, which aims to derive more fine-grained taxonomies of non-knowledge in SE.

#### References
[1] Bradley Efron. 1987. Better Bootstrap Confidence Intervals. Journal of the American Statistical Association. doi:10.1080/01621459.1987.10478410

[2] Nina Janich and Anne Simmerling. 2022. Linguistics and ignorance. In Routledge International Handbook of Ignorance Studies. Routledge. doi:10.4324/9781003100607

[3] György Szarvas, Veronika Vincze, Richárd Farkas, György Móra, and Iryna Gurevych. 2012. Cross-Genre and Cross-Domain Detection of Semantic Uncertainty. Computational Linguistics. doi:10.1162/coli_a_00098