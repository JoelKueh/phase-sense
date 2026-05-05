
cd ./dtwin
make
./bin/dtwin -p ./params.toml
mpv ./out/ds/0.avi
cd ..

read -p "Press ENTER to continue" </dev/tty
cd ./baseline
python ./baseline.py ../dtwin/out/ds/0.avi
cd ..

read -p "Press ENTER to continue" </dev/tty
cd ./tcn
python ./test-single.py
cd ..
