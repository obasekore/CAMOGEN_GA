import time
labels = []
while (1):
    label = labels[0]
    with open('output_from_background.txt', 'w') as fp:
        fp.write(str(time.time()) + ',' + label)
    time.sleep(0.1)
