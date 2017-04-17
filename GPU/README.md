# Notes for running the programs

- Run them 5-10 times and take the mean of the time because of the randomness of the initial centers. Also note the time per step for the same reason.

- Best block size and grid_size:

| Testcase      | Block Size  | Grid Size  |
|:-------------:|:-----------:|:----------:|
| road_spatial  | 256         | 1024       |
| daily_sports  | 64          | 4096       |

- Check for multiple values of k. k must be changed inside the input file.