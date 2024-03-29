# Introduction
This repository contains various artifacts, such as source code, experimental results, and other materials, that supplement our work on **Substructure-aware Log Anomaly Detection**.
SLAD framework is listed below.
![SLAD framework diagram](https://github.com/zzxhelloworld/SLAD/blob/main/Artifact/slad-framework.png)
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

   

