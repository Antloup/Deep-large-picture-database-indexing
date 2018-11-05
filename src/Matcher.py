from threading import Thread

import torchvision
from torchvision import transforms

from src.Net import Net
from src.Match import Match
import torch

class Matcher:



    def __init__(self, image, sampleSize, offset, threshold):

        self.net = Net()
        self.net.load_state_dict(torch.load("../models/model.pt"))
        self.net.eval()

        originalSize = image.size

        sampleSize = (36, 36)
        offset = (10, 10)

        resizedHeight = originalSize[1] - ((originalSize[1] - sampleSize[1]) % offset[1])

        self.matches = []

        while resizedHeight >= sampleSize[1]:

            print(resizedHeight, sampleSize[1])

            resizedWidth = round(originalSize[0] * resizedHeight / originalSize[1])

            resizedImage = torchvision.transforms.Resize((resizedHeight, resizedWidth))(image)

            topOffset = 0
            while (topOffset + sampleSize[1]) < resizedHeight:

                leftOffset = 0
                while (leftOffset + sampleSize[0]) < resizedWidth:
                    sampleImage = torchvision.transforms.functional.crop(resizedImage, topOffset, leftOffset,
                                                                         sampleSize[1],
                                                                         sampleSize[0])

                    is_face_probabilty = self.is_face(sampleImage)

                    if is_face_probabilty > threshold:
                        topOffsetOriginalSized = round(originalSize[1] * topOffset / resizedHeight)
                        leftOffsetOriginalSized = round(originalSize[0] * leftOffset / resizedWidth)

                        sampleHeightOriginalSized = round(originalSize[1] * sampleSize[1] / resizedHeight)
                        sampleWidthOriginalSized = round(originalSize[0] * sampleSize[0] / resizedWidth)

                        self.matches.append(Match(leftOffsetOriginalSized, topOffsetOriginalSized, sampleWidthOriginalSized,
                                             sampleHeightOriginalSized, is_face_probabilty))

                    leftOffset += offset[0]

                topOffset += offset[1]

            resizedHeight -= offset[1]

    def is_face(self, sampleImage):
        tensor = Net.transform(sampleImage)
        result_net = self.net(tensor.reshape(1, 1, 36, 36))
        predicted = result_net.detach().numpy()
        result = predicted.item(1)

        return result