## LLL for DNA
This repository aims to explore the use of large language models (LLMs) for analyzing DNA sequences using various available models.

More information about the use of LLM in genomic https://slides-ai-dna-eff0eb.pages.in2p3.fr/#/title-slide




## Test de Grover 
(add information on grover here)
Pour tester Grover j'ai decider de partie du notebook tutorial dispoinlbe ici https://zenodo.org/records/13315363 en le modifiant legement: à la place des CTCF motif, je vais essayer de detecter si une sequence est potentielement cancereuse ou pas. 
Impossible de lancer le notebookdirectement sur un noed de jeanzay car les nodes de clacul n'ont pas acces à internet , il a donc ete necessaire de scinder en 2 partie : une partie pre traitement (avec telechargement des données et creatation du dataset) et une partie traintement qui correspond au fine tunning 

creation d'un environnement conda 
conda env create -f env.yml  
conda activate dna_llm_env

