
read -p </dev/tty
cd ./dtwin
make
./bin/dtwin -p ./params.toml
mpv ./out/ds/0.avi
cd ..

read -p </dev/tty
python ./baseline/baseline.py dtwin/out/ds/0.avi

read -p </dev/tty
cd ./tcn
python ./test-single.py
cd ..
