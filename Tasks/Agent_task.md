Goal: Add unit tests stage for a project template, described below, using PyTest

Context: Project template is inside @Tasks/src/* directory. Inside @Tasks/src/code_databricks/databricks_notebooks/*.py and @Tasks/src/code_databricks/databricks_jobs/*.py there are python files.

Constraints:
 - keep test coverage > 85% in @Tasks/src/code_databricks/*
 - add all required python dependencies into @Tasks/src/code_databricks/requirements.txt
 - no lint errors
 - unit tests python files should be stored within @Tasks/src/code_databricks/tests/*
 - generate test report in junit xml format

Work in @Tasks/src directory only. Before implementation, generate a Plan for my approval