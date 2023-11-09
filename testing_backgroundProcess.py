import subprocess

pid = subprocess.Popen(['C:\Users\OBAS\Anaconda3\envs\py4web\python.exe', 'mainBackground.py'])

cnt = 0

while(cnt<1_000_000):
    print(cnt)
    with open('output_from_background.txt', 'w') as fp:
        txt = fp.read()
        if(txt != ''):
            tim,_ = txt.split(',')
            # str(time.time()) + ',' label

pid.kill()