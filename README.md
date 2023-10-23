# PyTorch_TensorHolo

# Install the requirements
pip install -r requirements.txt

# Dataset download
You can download the MIT-CGH-4K dataset from [This link] (https://drive.google.com/drive/folders/1TYDNfrfkehAJiUpDLadjxJzjDdgvC-GT).
After downloading the data, uncompress each ZIP file and put the subfolders into a directory, then modify the ```data_root``` in common.py 

# Start training the model
You can start training the model by running the following command:
``` 
sh train.sh
```

# Test the pretrained checkpoint
``` 
sh test.sh
```