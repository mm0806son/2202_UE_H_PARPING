# TP-TLP

Team Member:

- Zijie NING @[mm0806son](https://github.com/mm0806son)
- Lei WANG @[wang-lei-cn](https://github.com/wang-lei-cn)

## Exercise I: Warm up

### Question I.1 Before running this code, try to predict the following:

- What happens when OMP NUM THREADS = 1?
  We should see `Hello World... from thread 0 / 1`

- What happens when OMP NUM THREADS > 1?
  We should see `OMP_NUM_THREADS` of lines with Hello World, and during the test we found that they are mixed together.

- In what order will the different lines be printed when OMP NUM THREADS > 1?
  In a random order.

- Are you anticipating any issues with this code that would cause it to execute differently from expected (think about sharing)?
  The program will execute more slowly if these threads have shared resource.

## Exercise II: Parfor

Use `./exo_2 400000000 1 10 0 10` to execute.

### Question II.1 First, analyze the code:

- What does it do? What do the values stored in the vector correspond to?

  It's a transformation on the signal. The vector stores the results of the calculations corresponding to the different values of the variables.

- Can it be efficiently parallelized? If so, why? With what technology?

  Yes, because they can be executed separately and they don't have shared resources or ordonnance.

- Can issues similar to Exercise 1 be encountered when parallelizing this code?

  The random order of task completion will still occur, but will not pose a problem. The case of shared resources will not occur because all calculations are independent.

### Question II.2 Use 2 and 3 OpenMP tasks to accelerate the computation of the vector.

- Is the speedup as expected?

  Computed signal in 1 task: 12.734s
  Computed signal in 2 tasks: 6.86812s
  Computed signal in 3 tasks: 5.0251s

  We expected the speedup is linear, however is is not as expected. When task number changed from 1 to 2, the speed is about 2 times faster. But when the task number changed from 2 to 3, the computation is only a little faster.

- Analyze the speedup in function of the number of tasks, threads, and number of tasks per thread!

  When we increase the number of tasks, the calculation speed will increase. 
  If we increase the number of tasks, the speed will increase linearly until the number of tasks per thread is 1. In case the number of tasks per thread is 1, increasing the number of tasks has no noticeable performance gain.

- Is the whole vector initialized correctly for several values (even, odd, . . . ) of num_samples? Check the range of the plot to make sure everything is fine! If there is an issue, why and how can you fix it?

  We didn't see any issue.


### Question II.3 Compare the two parallelisation schemes you have implemented (tasks and #pragma omp for).

How do they differ in exibility, ease of use, . . . ?

In this problem, the `ParFor` method is easier with less code, and convenient to change the number of threads. The `Tasks` method requires manual creation of threads, which results in a lot of repetitive work. However, if we encounter a situation where we need to manually define different tasks for threads, the `Tasks` method will bring a higher level of freedom.

### Question II.4 Check the efficiency of the parallelization.

- How does the number of threads impact the speed of the program? (Check if we see the effect of hyper-threading).

  We changed the number of threads as follows and observed their speed.

  1 -> 12.26s
  2 -> 6.49s
  3 -> 5.12s
  4 -> 5.23s
  8 -> 4.67s
  32 -> 4.23s
  256 -> 3.96s

- What happens with a number of OpenMP threads up to the number of cores?

  We observed almost a proportional reduction in time.

- What happens when you have more OpenMP threads than hardware threads?

  The time spent was shortened, but not significantly.

## Exercise III: Fierce competition

### Question III.1 Using the code provided, verify that

- everything is working as expected without parallelization

  The program achieves the intended function and obtains the correct result.

- everything is working as expected with parallelization

  The program achieves the intended function and obtains the correct result. But the speed was not increased at all. 

### Question III.2 Using partial sums for each threads

The problem above is due to the shared variable, so using partial sums should solve this.