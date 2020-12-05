from torch import nn
from torchvision import models

class Autoencoder(nn.Module):

    def __init__(self):
        super (Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            #1x896x896
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, stride = 4, padding = 2),
            nn.ELU(),
            #1X224X224
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0),
            #1x224x224
            )

        self.decoder = nn.Sequential(
            #1x224x224
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
            #1x896x896
            )
        
        

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        x = Relu1.apply(x)
        
        return x


class Relu1(Function):

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        #print("fwd:", input[0])
        return input.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0]*=0.0
        grad_input[input>1]*=0.0

        return grad_input



class AECNN(nn.Module):

    def __init__(self, args, classCount):
        super (AECNN, self).__init__()

        self.classCount = classCount
        # self.y2 = torch.Tensor(bs, 3, h, w).cuda()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.encoder = nn.Sequential(
            #1x896x896
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, stride = 4, padding = 2),
            nn.ELU(),
            #1X224X224
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0),
            #1x224x224
            )

        self.decoder = nn.Sequential(
            #1x224x224
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(4),
            #1x896x896
            )


        #CLASSIFIER
        self.classifier = DenseNet121(classCount = self.classCount, isTrained = True, args = args)

    def forward(self, x):
        
        y = self.encoder(x)
        y = Relu1.apply(y)
        
        z1 = self.decoder(y)
        z1 = Relu1.apply(z1)
        
        bs, c, h, w = y.shape
        y2 = torch.Tensor(bs, 3, h, w).cuda()
        
        for img_no in range(bs):
            y2[img_no] = y[img_no]
            y2[img_no] = self.normalize(y2[img_no]) #broadcasting 1 channel to 3 channels

        z2 = self.classifier(y2)

        return z1, z2

def DenseNet121(classCount, isTrained, args):
    model = models.densenet121(pretrained=True)
    in_feat = model.classifier.in_features
    model.classifier = nn.Linear(in_feat, classCount)
    if isTrained:
        model.load_state_dict(torch.load(args.ckpt_path))
        return model

    return model