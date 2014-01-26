# CUFFT

# Notes on memory

For those who dive into the internals, one potentially-confusing point is that C's (or FFTW's) convention for representing array dimensions is opposite that of Julia. C's convention stems from the static representation of arrays,

```
const NX = 3
const NY = 5
double *myarray[NX][NY] = {
  {1.0, 2.0, 3.0, 4.0, 5.0},
  {6.0, 7.0, 8.0, 9.0, 10.0},
  {11.0, 12.0, 13.0, 14.0, 15.0}};
```

Consequently, `NX` represents the number of rows, and `NY` the number of columns (even though visually `x` is the horizontal axis and `y` the vertical axis). The first dimension therefore does _not_ correspond to the "fast" dimension in linear-memory layout.
