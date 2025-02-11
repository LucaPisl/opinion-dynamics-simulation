# Opinion Dynamics Simulation
This research explores how people's opinions shift and spread through social networks using a computer simulation technique called Agent-Based Modeling. By creating a dynamic model that mimics real-world social interactions, we investigate how individuals influence each other's perspectives. The simulation examines how opinions transform over time, tracking processes like consensus-building, growing disagreements, increasing polarization, and the conditions that might trigger unexpected social upheavals.

## Requirements

You need to install some python libraries using python pip in order to satisfy the imports in the program.

`pip install pandas numpy seaborn matplotlib networkx mesa plotly scipy nbformat`

If you cannot use pip to install python packages globally, create a python virtual environment (venv) and run the notebook with that. Tutorials are available online.

## Additional information

### What's working
- Agents are created with proper values
- Interactions do happen between agets, opinions do change
- Basically everything at ABM init stage works
- Data visualization with charts
- Exporting to Gephi / other tools for network visualization
- Essentially, code executes


### What's somewhat working
- Dynamic linking creation/breaking -> currently link creation and destrucion seems sporradic, it's 100% related to the variables, but also the step function is very complicated and probably contains some bugs
- Tweaking the global variables to get actual accurate bevahior

### What NEEDS improvement
- OPTIMIZATION -> right now it's very slow, it takes

### What could use improvement, low priority
- Reworking of the way the code uses the mesa scheduler, specifically the random scheduler (/scrambler I guess), because it's going to be deprecated

### What's not working
- Advanced network formation -> Currently networks are formed in a "free form" let's say, connections just form (based on the requirements), there are no actual clusters (communities), many agents are connected to each other and it's hard to actually tell what's happening.



### Key takeaway
- Parallelization, multi-threading, etc. is basically required for this to actually be feasible
