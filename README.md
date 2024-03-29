# Introduction
This repository contains various artifacts, such as source code, experimental results, and other materials, that supplement our work on **Substructure-aware Log Anomaly Detection**.\
&nbsp;&nbsp;&nbsp;&nbsp;Foremost, the repository contains an implementation of the lossless, dependency-preserving decomposition algorithm that minimizes the key number of subschemata in BCNF and the non-key FD number of subschemata in 3NF. Variant A1: <kbd>src/nf/iCONFOpt_minf_maxk.java</kbd> breaks further ties between redundant 3NF subschemata (in case they have the same number of non-key FDs) by prioritizing those with a higher number of minimal keys, while variant A2: <kbd>src/nf/iCONF.java</kbd>) only breaks ties using the number of non-key FDs. We have also implemented an additional three algorithms that allow us to compare our algorithms with the previous state-of-the-art (SOTA) algorithm CONF: A3(<kbd>src/nf/CONF.java</kbd>), as well as BC-Cover: A4(<kbd>src/nf/DecompAlg2.java</kbd>) and Synthesis: A5(<kbd>src/nf/DecompAlg4.java</kbd>). We have also included the code of implementations for other experiments: <kbd>src/exp/</kbd> and <kbd>src/util/</kbd>. For all experimental results, logs and some sql scripts are included, too (<kbd>Artifact/Experiments/</kbd>). In the following sections, we describe how our experiments can be reproduced. 
# Preliminaries: Getting databases ready for experiments
> 1. Import 12 datasets as SQL databases
>> We have used MySQL 8.0.29 as database workbench. Firstly, please create a database. Afterwards, import the [12 datasets](https://hpi.de/naumann/projects/repeatability/data-profiling/fds.html) as MySQL databases by setting column names as 0,1,...,n-1 where n is the number of columns in a given dataset. In addition, please create a column named "id" as an auto_increment attribute for each table that will facilitate the removal of updated tuples quickly.
> 2. Import TPC-H benchmark
>> Please visit the [website](https://relational.fit.cvut.cz/dataset/TPCH) and export the TPC-H database as an .sql file. Then, please import the file in your own local MySQL workbench. Under <kbd>Artifact/Experiments/TPCH/</kbd> we have included all 22 official SQL queries and refresh functions for use in our experiments.
# How to run code from the command line
1. Clone the repository:
   ```bash
   git clone https://github.com/zzxhelloworld/iCONF.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your_project_directory
   ```
3. Run separate code from the command line for experiments:
   
   3.1 Mini Study
   ```bash
   javac SyntheticExpForCaseStudy.java
   java SyntheticExpForCaseStudy <output_path> <db_table_name> <experiment_repeat_num> <synthetic_dataset_num> <insert_num>
   ```
   3.2 Why do we parameterize normalization?
   ```bash
   javac TPCHWorkloadExp.java
   java TPCHWorkloadExp <experiment_repeat_num> <TPCH_sql_path> <TPCH_schema_output_path> <experimental_result_output_path>
   ```
   3.3 How good are our algorithms?
   ```bash
   javac DecompExp.java
   java DecompExp <dataset_name> <experimental_results_output_directory>
   ```
   3.4 How much overhead do we save?
   ```bash
   javac SubschemaPerfExp.java
   java SubschemaPerfExp <experiment_repeat_num> <schema_sample_num> <experimental_results_output_path>
                         <decomposition_algs_separated_by_commas> <dataset_name> <experimental_results_output_directory>
   ```

   

