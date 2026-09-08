import torch.nn as nn


class Space1Space2FrequencyTimeCNN(nn.Module):

   def __init__(self):

      super().__init__()

      # (batch, 9, 65, 21, 21)

      self.conv1 = nn.Conv3d(
         in_channels=9,
         out_channels=6,
         kernel_size=3,
         padding=1
      )
      # (batch, 6, 65, 21, 21)


      self.relu1 = nn.ReLU()


      self.pool1 = nn.MaxPool3d(
         kernel_size=(2, 2, 2)
      )
      # (batch, 6, 32, 10, 10)


      self.conv2 = nn.Conv3d(
         in_channels=6,
         out_channels=4,
         kernel_size=3,
         padding=1
      )
      # (batch, 4, 32, 10, 10)

      self.relu2 = nn.ReLU()

      self.pool2 = nn.MaxPool3d(
         kernel_size=(2, 2, 2)
      )
      # (batch, 4, 16, 5, 5)

      self.flatten = nn.Flatten() #4x16x5x5 = 1600

      self.classifier = nn.Linear(
         4 * 16 * 5 * 5,
         4
      )

   def extract_features(self, x):


      x = self.conv1(x)
      x = self.relu1(x)
      x = self.pool1(x)

      x = self.conv2(x)
      x = self.relu2(x)
      x = self.pool2(x)

      x = self.flatten(x)

      return x

   def forward(self, x):

      x = self.extract_features(x)
       
      x = self.classifier(x)


      return x



class Space1Space2TimeFrequencyCNN(nn.Module):

   def __init__(self):

      super().__init__()

      # (batch, 65, 9, 21, 21)

      self.conv1 = nn.Conv3d(
         in_channels=65,
         out_channels=32,
         kernel_size=3,
         padding=1
      )
      # (batch, 32, 9, 21, 21)


      self.relu1 = nn.ReLU()


      self.pool1 = nn.MaxPool3d(
         kernel_size=(2, 2, 2)
      )
      # (batch, 32, 4, 10, 10)


      self.conv2 = nn.Conv3d(
         in_channels=32,
         out_channels=16,
         kernel_size=3,
         padding=1
      )
      # (batch, 16, 4, 10, 10)

      self.relu2 = nn.ReLU()

      self.pool2 = nn.MaxPool3d(
         kernel_size=(2, 2, 2)
      )
      # (batch, 16, 2, 5, 5) 

      self.flatten = nn.Flatten() #16x2x5x5 = 800

      self.classifier = nn.Linear(
         16 * 2 * 5 * 5, 
         4
      )

   def extract_features(self, x):

      x = self.conv1(x)
      x = self.relu1(x)
      x = self.pool1(x)


      x = self.conv2(x)
      x = self.relu2(x)
      x = self.pool2(x)


      x = self.flatten(x)


      return x

   def forward(self, x):

      x = self.extract_features(x)

      x = self.classifier(x)

      return x



class TimeFrequencySpace2Space1CNN(nn.Module):

   def __init__(self):


        super().__init__()

       # (batch, 21, 21, 65, 9)

        self.conv1 = nn.Conv3d(
           in_channels=21,
           out_channels=10,
           kernel_size=3,
           padding=1
        )
        # (batch, 10, 21, 65, 9)


        self.relu1 = nn.ReLU()


        self.pool1 = nn.MaxPool3d(
           kernel_size=(2, 2, 2)
        )
        # (batch, 10, 10, 32, 4)


        self.conv2 = nn.Conv3d(
           in_channels=10,
           out_channels=5,
           kernel_size=3,
           padding=1
         )
        # (batch, 5, 10, 32, 4)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool3d(
           kernel_size=(2, 2, 2)
        )
        # (batch, 5, 5, 16, 2)

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(
            5 * 5 * 16 * 2,
            4
        )




   def extract_features(self, x):


        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)


        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)


        x = self.flatten(x)


        return x




   def forward(self, x):


       x = self.conv1(x)
       x = self.relu1(x)
       x = self.pool1(x)


       x = self.conv2(x)
       x = self.relu2(x)
       x = self.pool2(x)


       x = self.flatten(x)


       x = self.classifier(x)


       return x



class TimeFrequencySpace1Space2CNN(nn.Module):

   def __init__(self):


        super().__init__()

       # (batch, 21, 21, 65, 9)

        self.conv1 = nn.Conv3d(
           in_channels=21,
           out_channels=10,
           kernel_size=3,
           padding=1
        )
        # (batch, 10, 21, 65, 9)


        self.relu1 = nn.ReLU()


        self.pool1 = nn.MaxPool3d(
           kernel_size=(2, 2, 2)
        )
        # (batch, 10, 10, 32, 4)


        self.conv2 = nn.Conv3d(
           in_channels=10,
           out_channels=5,
           kernel_size=3,
           padding=1
         )
        # (batch, 5, 10, 32, 4)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool3d(
           kernel_size=(2, 2, 2)
        )
        # (batch, 5, 5, 16, 2)

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(
            5 * 5 * 16 * 2,
            4
        )




   def extract_features(self, x):


        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)


        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)


        x = self.flatten(x)


        return x




   def forward(self, x):


       x = self.conv1(x)
       x = self.relu1(x)
       x = self.pool1(x)


       x = self.conv2(x)
       x = self.relu2(x)
       x = self.pool2(x)


       x = self.flatten(x)


       x = self.classifier(x)


       return x


