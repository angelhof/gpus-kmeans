# Notes for running the programs

- Run them 5-10 times and take the mean of the time because of the randomness of the initial centers. Also note the time per step for the same reason.

- Best values for block sizes and grid sizes:

| Testcase       | Block Size  | Grid Size  |
|:--------------:|:-----------:|:----------:|
| road_spatial   | 256         | 1024       |
| daily_sports   | 64          | 4096       |
| minebench/edge | 64          | 1024       |

- Check for multiple values of k. k must be changed inside the input file.

- For minebench use k >= 200.