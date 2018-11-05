import torch
import torchvision
from torchvision import transforms

from PIL import Image, ImageDraw
from Net import Net
from Match import Match

if __name__ == '__main__':

    import random

    net = Net()
    net.load_state_dict(torch.load("../model.pt"))
    net.eval()

    random.seed(20)

    threshold = 0.95


    # Mock of face detector
    def is_face(sampleImage):

        # sampleImage.show()

        tensor = Net.transform(sampleImage)
        result_net = net(tensor.reshape(1, 1, 36, 36))
        predicted = result_net.detach().numpy()
        result = predicted.item(1)
        # result = predicted[0].numpy().item(1)
        # print(result_net)
        # print(result)

        # if result == 1:
            # print(result_net)
            # sampleImage.show()
            # input()

        return result



    image = Image.open("../test.jpg")

    # image.show()

    # Resize the image and crop it at different levels

    originalSize = image.size

    sampleSize = (36, 36)
    offset = (10, 10)

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

                if is_face_probabilty > .98:
                    topOffsetOriginalSized = round(originalSize[1] * topOffset / resizedHeight)
                    leftOffsetOriginalSized = round(originalSize[0] * leftOffset / resizedWidth)

                    sampleHeightOriginalSized = round(originalSize[1] * sampleSize[1] / resizedHeight)
                    sampleWidthOriginalSized = round(originalSize[0] * sampleSize[0] / resizedWidth)

                    matches.append(Match(leftOffsetOriginalSized, topOffsetOriginalSized, sampleWidthOriginalSized,
                                              sampleHeightOriginalSized, is_face_probabilty))

                leftOffset += offset[0]

            topOffset += offset[1]

        resizedHeight -= offset[1]

    bestMatches = []

    # Filter the best matches

    matchIndex = 0

    for match in matches:

        bestMatchIndex = 0

        nonEmptyIntersectedBestMatchIndexes = []

        for bestMatch in bestMatches:

            # If the intersection is not empty
            if bestMatch.right() > match.left and \
                    match.right() > bestMatch.left and \
                    bestMatch.bottom() > match.top and \
                    match.bottom() > bestMatch.top:
                nonEmptyIntersectedBestMatchIndexes.append(bestMatchIndex)

            bestMatchIndex += 1

        if len(nonEmptyIntersectedBestMatchIndexes) == 0:
            bestMatches.append(match)
        else:

            betterProbability = False

            for bestMatchIndex in nonEmptyIntersectedBestMatchIndexes:
                if match.probability > bestMatches[bestMatchIndex].probability:
                    betterProbability = True
                    break

            if betterProbability:
                for bestMatchIndex in nonEmptyIntersectedBestMatchIndexes:
                    bestMatches[bestMatchIndex] = match

        matchIndex += 1

    bestMatches = list(set(bestMatches))  # Remove duplicates

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

    for bestMatch in matches:
        normalizedProbability = bestMatch.probability
        color = compute_color(normalizedProbability)
        # color = (0,0,0)
        width = 2

        draw.line((bestMatch.left, bestMatch.top, bestMatch.right(), bestMatch.top), fill=color, width=width)  # top
        draw.line((bestMatch.left, bestMatch.bottom(), bestMatch.right(), bestMatch.bottom()), fill=color,
                  width=width)  # bottom
        draw.line((bestMatch.left, bestMatch.top, bestMatch.left, bestMatch.bottom()), fill=color, width=width)  # left
        draw.line((bestMatch.right(), bestMatch.top, bestMatch.right(), bestMatch.bottom()), fill=color,
                  width=width)  # right
        draw.line((bestMatch.left, bestMatch.top, bestMatch.right() + 1, bestMatch.top), fill=color, width=width)  # top

        x = round((bestMatch.left + bestMatch.right()) / 2)
        y = round((bestMatch.top + bestMatch.bottom()) / 2)

        # color = (255, 0, 0)
        draw.line((x, y, x+1, y+1), fill=color, width=width)  # center

        draw.line((bestMatch.left, bestMatch.top + 5, bestMatch.right() + 1, bestMatch.top + 5), fill=color,
                  width=14)  # top

        draw.text((bestMatch.left + 2, bestMatch.top), str(normalizedProbability)[0:7], (255, 255, 255))

    image.show()
