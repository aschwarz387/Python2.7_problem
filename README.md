# Reinforcement learning Python3.6 to Python2.7 conversion problems

## Note: this was done on Ubuntu16.04 system

*based on Teach a Quadcopter How to Fly!*

In this project, you will design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm of your choice! 

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.

```
git clone https://github.com/udacity/RL-Quadcopter-2.git
cd RL-Quadcopter-2
```

2. Create and activate a new environment.

```
conda create -n quadcop python=3.6 matplotlib numpy pandas
source activate quadcop
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `quadcop` environment. 
```
python -m ipykernel install --user --name quadcop --display-name "quadcop"
```

4. Open the notebook.
```
jupyter notebook 

- I have included 2 notebooks 
    - Quadcopter_Project_Py36.ipynb  is for use with Python3.6 and works completely
    - Quadcopter_Project_Py27.ipynb  is my attempt to convert code to work with python2.7 but it fails


