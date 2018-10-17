import torchvision
from PIL import Image, ImageDraw

import random


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

    def __eq__(self, other):
        return self.left == other.left and\
            self.top == other.top and\
            self.width == other.width and\
            self.height == other.height

    def __hash__(self):
        return hash((self.left, self.top, self.width, self.height))

random.seed(20)

threshold = 0.999


# Mock of face detector
def is_face():
    probabilty = random.randint(0, 100000) / 100000

    if threshold < probabilty < 1:
        return probabilty

    return 0


image = Image.open("test.jpg")

# image.show()

# Resize the image and crop it at different levels

originalSize = image.size

sampleSize = (32, 32)
offset = (10, 10)

resizedHeight = originalSize[1] - ((originalSize[1] - sampleSize[1]) % offset[1])

matches = []

while resizedHeight >= sampleSize[1]:

    resizedWidth = round(originalSize[0] * resizedHeight / originalSize[1])

    resizedImage = torchvision.transforms.Resize((resizedHeight, resizedWidth))(image)

    topOffset = 0
    while (topOffset + sampleSize[1]) < resizedHeight:

        leftOffset = 0
        while (leftOffset + sampleSize[0]) < resizedWidth:
            sampleImage = torchvision.transforms.functional.crop(resizedImage, topOffset, leftOffset, sampleSize[1],
                                                                 sampleSize[0])

            is_face_probabilty = is_face()

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

    draw.line((bestMatch.left, bestMatch.top + 5, bestMatch.right() + 1, bestMatch.top + 5), fill=color, width=14)  # top

    draw.text((bestMatch.left + 2, bestMatch.top), str(bestMatch.probability)[1:], (255, 255, 255))

image.show()

