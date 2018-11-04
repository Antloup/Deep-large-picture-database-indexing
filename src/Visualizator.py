import torch
import torchvision

from PIL import Image, ImageDraw
from Net import Net
# from IntersectCluster import IntersectCluster
from MagnetCluster import MagnetCluster

if __name__ == '__main__':

    import random

    net = Net()
    net.load_state_dict(torch.load("../net_model.pt"))
    net.eval()


    class matchClass:
        def __init__(self, left, top, width, height, probability):
            self.left = left
            self.top = top
            self.width = width
            self.height = height
            self.probability = probability

        def right(self):
            return self.left + self.width

        def bottom(self):
            return self.top + self.height

        def center(self):
            return [self.left + self.width / 2, self.top + self.height/2]

        def __eq__(self, other):
            return self.left == other.left and \
                   self.top == other.top and \
                   self.width == other.width and \
                   self.height == other.height

        def __hash__(self):
            return hash((self.left, self.top, self.width, self.height))


    random.seed(20)

    threshold = 0.99


    # Mock of face detector
    def is_face(sampleImage):
        probabilty = random.randint(0, 100000) / 100000

        if threshold < probabilty < 1:
            return probabilty

        return 0

        grayscale = sampleImage.convert('L')
        tensor = torchvision.transforms.ToTensor()(grayscale)
        _, predicted = torch.max(net(tensor.reshape(1, 1, 36, 36)), 1)

        if predicted[0] == 1:
            return 1
        else:
            return 0

        return predicted[0] == 1


    image = Image.open("../test.jpg")

    # image.show()

    # Resize the image and crop it at different levels

    originalSize = image.size

    sampleSize = (36, 36)
    offset = (20, 20)

    resizedHeight = originalSize[1] - ((originalSize[1] - sampleSize[1]) % offset[1])

    matches = []

    while resizedHeight >= sampleSize[1]:

        print(resizedHeight, sampleSize[1])

        resizedWidth = round(originalSize[0] * resizedHeight / originalSize[1])

        resizedImage = torchvision.transforms.Resize((resizedHeight, resizedWidth))(image)

        topOffset = 0
        while (topOffset + sampleSize[1]) < resizedHeight:

            leftOffset = 0
            while (leftOffset + sampleSize[0]) < resizedWidth:
                sampleImage = torchvision.transforms.functional.crop(resizedImage, topOffset, leftOffset, sampleSize[1],
                                                                     sampleSize[0])

                is_face_probabilty = is_face(sampleImage)

                if is_face_probabilty > 0:
                    topOffsetOriginalSized = round(originalSize[1] * topOffset / resizedHeight)
                    leftOffsetOriginalSized = round(originalSize[0] * leftOffset / resizedWidth)

                    sampleHeightOriginalSized = round(originalSize[1] * sampleSize[1] / resizedHeight)
                    sampleWidthOriginalSized = round(originalSize[0] * sampleSize[0] / resizedWidth)

                    matches.append(matchClass(leftOffsetOriginalSized, topOffsetOriginalSized, sampleWidthOriginalSized,
                                              sampleHeightOriginalSized, is_face_probabilty))

                leftOffset += offset[0]

            topOffset += offset[1]

        resizedHeight -= offset[1]


    # Filter the best matches

    bestMatches = MagnetCluster.extract(matches, 60)

    print(len(bestMatches))


    # Return a color between red 0%, yellow 50% and green 100%
    def compute_color(probability):
        if probability < 0.5:
            q = probability / 0.50
            r = round(0xDB + q * (0xFB - 0xDB))
            g = round(0x28 + q * (0xBD - 0x28))
            b = round(0x28 + q * (0x08 - 0x28))
        else:
            q = (probability - 0.50) / 0.75
            r = round(0xFB + q * (0x21 - 0xFB))
            g = round(0xBD + q * (0xBA - 0xBD))
            b = round(0x08 + q * (0x45 - 0x08))
        return r, g, b


    # Draw best matches

    draw = ImageDraw.Draw(image)

    for bestMatch in bestMatches:
        normalizedProbability = (bestMatch.probability - threshold) / (1 - threshold)
        color = compute_color(normalizedProbability)
        width = 2

        draw.line((bestMatch.left, bestMatch.top, bestMatch.right(), bestMatch.top), fill=color, width=width)  # top
        draw.line((bestMatch.left, bestMatch.bottom(), bestMatch.right(), bestMatch.bottom()), fill=color,
                  width=width)  # bottom
        draw.line((bestMatch.left, bestMatch.top, bestMatch.left, bestMatch.bottom()), fill=color, width=width)  # left
        draw.line((bestMatch.right(), bestMatch.top, bestMatch.right(), bestMatch.bottom()), fill=color,
                  width=width)  # right
        draw.line((bestMatch.left, bestMatch.top, bestMatch.right() + 1, bestMatch.top), fill=color, width=14)  # top

        draw.line((bestMatch.left, bestMatch.top + 5, bestMatch.right() + 1, bestMatch.top + 5), fill=color,
                  width=14)  # top

        draw.text((bestMatch.left + 2, bestMatch.top), str(bestMatch.probability)[1:], (255, 255, 255))

    image.show()
