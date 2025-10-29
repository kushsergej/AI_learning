Goal: Add unit tests stage for a project template, described below, using PyTest

Context: Project template is inside @Tasks/src/* directory. Inside @Tasks/src/databricks_notebooks/*.py and @Tasks/src/databricks_jobs/*.py there are python files.

Constraints:
 - keep test coverage > 85% in @Tasks/src/*
 - add all required python dependencies into requirements.txt
 - no lint errors
 - test python files should be stored within @Tasks/code_databricks/tests/*
 - generate test report in junit xml format

Work in @src directory only. Before implementation, generate a Plan for my approval