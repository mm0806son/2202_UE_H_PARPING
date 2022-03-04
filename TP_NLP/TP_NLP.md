# TP_NLP

mpirun -np 4 -host localhost,pc-elec-187,pc-elec-188,pc-elec-189 main_1

## main_2

echo -e -n "hello" | sha512sum | cut -f1 -d" "
9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca72323c3d99ba5c11d7c7acc6e14b8c5da0c4663475c2e5c3adef46f73bcdec043

mpirun -np 4 -host localhost,pc-elec-187,pc-elec-188,pc-elec-189 main_2 /usr/share/dict/words 10000 9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca72323c3d99ba5c11d7c7acc6e14b8c5da0c4663475c2e5c3adef46f73bcdec043

echo -e -n "hello" | sha512sum | cut -f1 -d" "
