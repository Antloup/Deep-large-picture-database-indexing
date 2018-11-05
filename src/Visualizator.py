import torchvision

from PIL import Image, ImageDraw

from src.MagnetCluster import MagnetCluster
from src.Matcher import Matcher

if __name__ == '__main__':

    filename = "test.jpg"

    image = Image.open("../samples/" + filename)

    maximalHeight = 1000

    if maximalHeight != 0 and image.size[1] > maximalHeight:
        preprocessedWidth = int(maximalHeight * image.size[0] / image.size[1])
        image = torchvision.transforms.Resize((maximalHeight, preprocessedWidth))(image)

    matcher = Matcher(image, sampleSize=(36, 36), offset=(10, 10), threshold=0.99)

    matches = matcher.matches

    bestMatches = MagnetCluster.extract(matches, 20)

    print(str(len(bestMatches)) + " matches")

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
        return r, g, b, 192


    # Draw best matches

    image = image.convert('RGBA')
    layer = Image.new('RGBA', image.size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(layer)

    for bestMatch in bestMatches:
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
        draw.line((bestMatch.left, bestMatch.top, bestMatch.right() + 1, bestMatch.top), fill=color, width=width)  # text bg

        x = round((bestMatch.left + bestMatch.right()) / 2)
        y = round((bestMatch.top + bestMatch.bottom()) / 2)

        draw.line((x, y, x+1, y+1), fill=color, width=width)  # center point

        draw.line((bestMatch.left, bestMatch.top + 5, bestMatch.right() + 1, bestMatch.top + 5), fill=color,
                  width=14)  # top

        draw.text((bestMatch.left + 2, bestMatch.top), str(bestMatch.nbVotes) + " - " + str(normalizedProbability)[0:7], (255, 255, 255, 192))

    out = Image.alpha_composite(image, layer)

    out.show()

    out = out.convert('RGB')
    out.save("../results/result_" + filename)