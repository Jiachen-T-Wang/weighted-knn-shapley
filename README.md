# Command line:
```
python applications_knn.py --task collect_sv --dataset click --value_type fastWKNN-SV --n_data 50 --n_val 50 --flip_ratio 0.1 --K 5 --kernel rbf --dis_metric l2 --eps 0 --n_repeat 1 --n_bits 3 --temp 0.1
```

Explanation:
- `eps` controls the degree of approximation. `eps=0` is the exact algorithm. 
- `n_bits` is the number of bits for discretization. 