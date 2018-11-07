from threading import Thread

import torchvision
from torchvision import transforms

from src.Net import Net
from src.Match import Match
import torch
import time
import multiprocessing
from multiprocessing import Process, Queue


def f(args):

    tasks, results, doneTasks, workerId, matcher_task_to_solve = args

    print(doneTasks, workerId)

    net = Net()
    net.load_state_dict(torch.load("../models/model.pt"))
    net.eval()

    for task in tasks:

        sampleImage, sampleSize, originalSize, threshold, topOffset, leftOffset, resizedWidth, resizedHeight = task

        tensor = Net.transform(sampleImage)
        result_net = net(tensor.reshape(1, 1, 36, 36))
        predicted = result_net.detach().numpy()
        is_face_probabilty = predicted.item(1)

        if is_face_probabilty > threshold:
            topOffsetOriginalSized = round(originalSize[1] * topOffset / resizedHeight)
            leftOffsetOriginalSized = round(originalSize[0] * leftOffset / resizedWidth)

            sampleHeightOriginalSized = round(originalSize[1] * sampleSize[1] / resizedHeight)
            sampleWidthOriginalSized = round(originalSize[0] * sampleSize[0] / resizedWidth)

            results.put(Match(leftOffsetOriginalSized, topOffsetOriginalSized, sampleWidthOriginalSized,
                              sampleHeightOriginalSized, is_face_probabilty))

        doneTasks[workerId] += 1

        doneTasksCounter = 0
        for wId in range(len(doneTasks)):
            doneTasksCounter += doneTasks[wId]

        if doneTasksCounter % 1000 == 0:
            progress = int(doneTasksCounter / matcher_task_to_solve * 100)
            print(str(doneTasksCounter) + "/" + str(matcher_task_to_solve) + " tasks done (" + str(progress) + "%)")


class Matcher:



    def __init__(self, image, sampleSize, offset, threshold):

        originalSize = image.size

        sampleSize = (36, 36)
        offset = (10, 10)

        resizedHeight = originalSize[1] - ((originalSize[1] - sampleSize[1]) % offset[1])

        self.matches = []

        tasks = []

        print("Work is preparing...")

        while resizedHeight >= sampleSize[1]:

            resizedWidth = round(originalSize[0] * resizedHeight / originalSize[1])

            resizedImage = torchvision.transforms.Resize((resizedHeight, resizedWidth))(image)

            topOffset = 0
            while (topOffset + sampleSize[1]) < resizedHeight:

                leftOffset = 0
                while (leftOffset + sampleSize[0]) < resizedWidth:

                    sampleImage = torchvision.transforms.functional.crop(resizedImage, topOffset, leftOffset,
                                                                         sampleSize[1],
                                                                         sampleSize[0])

                    tasks.append((sampleImage, sampleSize, originalSize, threshold, topOffset, leftOffset, resizedWidth, resizedHeight))

                    leftOffset += offset[0]

                topOffset += offset[1]

            resizedHeight -= offset[1]

        matcher_task_to_solve = len(tasks)

        number_of_worker = multiprocessing.cpu_count()
        number_of_worker = 1

        threads = {}
        threadsTasks = []
        threadsDoneTasks = []
        threadsResults = []

        for t in range(number_of_worker):
            threadsTasks.append([])
            threadsDoneTasks.append(0)
            threadsResults.append(Queue())


        currentThreadIndex = 0
        while len(tasks) > 0:
            threadsTasks[currentThreadIndex].append(tasks.pop())
            currentThreadIndex = (currentThreadIndex + 1) % number_of_worker

        start = time.time()

        for t in range(number_of_worker):
            threads[t] = Process(target=f, args=([threadsTasks[t], threadsResults[t],threadsDoneTasks,t,matcher_task_to_solve],))
            threads[t].start()

        print("Work is progressing...")
        print(str(number_of_worker) + " worker(s)")
        print(str(matcher_task_to_solve) + " task(s)")

        for t in range(number_of_worker):
            threads[t].join()

            while not (threadsResults[t].empty()):
                self.matches.append(threadsResults[t].get())

     #
        # for t in range(number_of_worker):
        #     threads[t].join()
        #     while not(threadsResults[t].empty()):
        #         print(threadsResults[t].get())
        #
        end = time.time()

        duration = end - start

        print(str(matcher_task_to_solve) + "/" + str(matcher_task_to_solve) + " tasks done (100%)")

        print("Duration in seconds : " + str(int(duration)))
