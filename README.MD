# ML Model

The codes used to train the machine learning models are in the ML_Models directory.  

[slab_train.py](https://github.com/lmj1029123/SGCMC_Acrolein_AgPd/blob/main/ML_Models/slab_train.py) and [bulk_train.py](https://github.com/lmj1029123/SGCMC_Acrolein_AgPd/blob/main/ML_Models/bulk_train.py) are used to train the models for AgPd bulk and AgPd slab. They are essentially the same code but with different training dataset.  

[adsroption_model.py](https://github.com/lmj1029123/SGCMC_Acrolein_AgPd/blob/main/ML_Models/adsorption_model.py) is used to generate the model for adsortpion energy.

# SGCMC

The codes for running the semi-grand canonical Monte-Carlo simulation are in the SGCMC directory. 

[MC_bulk.py](https://github.com/lmj1029123/SGCMC_Acrolein_AgPd/blob/main/SGCMC/MC_bulk.py) is used to do MC simulation on the AgPd bulk to obtain the \Delta chemical potential.

[MC_slab.py](https://github.com/lmj1029123/SGCMC_Acrolein_AgPd/blob/main/SGCMC/MC_slab.py) is used to do MC simulation on the AgPd slab to study the surface segregation under vacuum.

[MC_Motiff.py](https://github.com/lmj1029123/SGCMC_Acrolein_AgPd/blob/main/SGCMC/MC_Motiff.py) is used to do MC simulation on the AgPd slab with acrolein.

For all these three scripts we do 

```
python MC_{*}.py "random seed" "initial number of Pd" "\Delta chemical potential" "T" "model seed" 
```

The MC simulation will output a log file and a json file. The json file contains a dictionary that specifies the configuration of the chemical system at each MC step.


For system with slab or bulk, there are 3 keys:  
- size: size of the slab.  
- Pd: index of Pd.  
- energy: total energy of the system.


For system with slab and acrolein, there are 9 keys:  
- size: size of the slab.  
- h : the height that the pseudo atom is placed on the sites.  
- cutoff: cutoff radius used for fingerprinting.  
- site_distance: the distance between pseudo atoms.  
- max_N_acrolein: maximum number of acrolein allowed on the surface.  
- 1_Pd: number of Pd on the first layer.  
- 2_Pd: number of Pd on the second layer.  
- Pd: index of Pd in the variable layers. E.g. index 1 is the first atom in the second layer.   
- site: index of the sites  



# Plotting

The scripts used to generate the plots in the paper are in the Plotting directory.
