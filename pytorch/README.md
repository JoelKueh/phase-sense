# Pytorch Pipeline

This is the Pytorch ML implementation for the phase-sense ML pipeline.

## Dependencies

This project relies on the usage of a few software resources:

1. Google Colab / Pytorch Installation

2. GPU processing capabilities (recommended G4 GPU or better)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ROCm avaliable? {torch.cuda.is_available()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}") 

3. A path to Google Drive with generated .avi and .meta files for training, testing, and evaluation of the model

    ` DATA_DIR = "../dtwin/out/ds/" `

## Usage 

### Initialization

Connect to GPU in Colab, verify Drive is connected
![alt text](image.png)

DATA_DIR should be set to path to folder in Drive with testing data

### Frame Extraction and Manipulation

Frames = 300
Image Size 224 x 224 pixels
Seq Len = 100

train test split 80/20

VideoDataset class is used to transform meta and avi files into usable arrays of frames

metadata implemented as a curve for training

### Network Implementation

#### Two Models 

#### CNN-TCN

ResNet50 backbone with two layers left for training

--> average pooling --> temporal filter with 2048 channels in and 64 out, window size of 5 --> Linear layer (64,1)

Relu

#### CNN-LSTM

ResNet50 backbone but fine tuned last block only

LSTM layer (2048) --> dropout --> linear

### Training

Training parameters 

    model = CNN_TCN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 

10 epochs were run to train the model

`  print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}") `

### Testing

Run test.py 

### Verification

### Example Output Plots

Run plot.py

Output from test.py ` ./results.csv ` is used to run this file

Outputs
1. Normalized Emergence Curve
2. Emergence Curve Onset Error
3. Two Way Minmax Emergence Curve Error
4. One Way Minmax Emergence Curve Error


## Contributing

## License

## Contact

Cole Jaeger - jaege320@umn.edu
Joel Kuehne - kuehn348@umn.edu

## Acknowledgement 

Prof. Sang-Hyun Oh at the University of Minnesota 
Department of Computer and Electrical Engineering

Contact: sang@umn.edu

Kyle Howey, Grad Student at the University of Minnesota
Department of Computer and Electrical Engineering

Contact: howey024@umn.edu
