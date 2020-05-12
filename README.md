# ConvVAE
Modular and reusable Conv VAE
Comes with script to make images of boxes to test model
Goals: 
- Modularity:
	- To use, should just have to instantiate model

	- Model object should have functionality for:
	- training ala Train(input data, target data, training time, batch size)
	- loading checkpoints ala Load(weight path)
	- saving checkpoints ala Save(weight path)
	- predicting ala Predict(input data)

	- Note this functionality is centered on data (a data translation tool may be needed later)

- Reusability
	- Model ctor should take layers wanted and size requirements 
	- Same model should be reusable for any VAE needs 	 
	  
	  
Requirements:   

- Pytorch, CUDA (optional)  

Usage:    

  - Check VAE.py for all methods and documentation
  

