<img src="https://user-images.githubusercontent.com/93063038/186162010-afc387b2-d664-4b05-a3ec-94866ff60cea.png" title="" alt="image" data-align="center">

> 虽然这题直接用简单的快排就能解决辣，但我看到了几种更巧妙的解法所以记录一下

该方法思路较为简单：

+ 当如一般性的快排一样结束while循环时，$p[0..j]$中的元素均满足$<=x$，因此，前$j+1$小的数都包含于其中，$k-1<=j$即$k<=j+1$，则此时$k$一定在$p[l..j]$之间。

**c++**

```cpp
#include <iostream>
#include <vector>

using namespace std;
vector<int> a;


int quick_sort(int l, int r, int k) {
    if(l >= r) return a[k];

    int x = a[l], i = l - 1, j = r + 1;
    while (i < j) {
        do i++; while (a[i] < x);
        do j--; while (a[j] > x);
        if (i < j) swap(a[i], a[j]);
    }
    if (k <= j) return quick_sort(l, j, k);
    else return quick_sort(j + 1, r, k);
}

int main() {
    int n, k;
    cin >> n >> k;
    a = vector<int>(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }

    cout << quick_sort(0, n - 1, k - 1) << endl;

    return 0;
}
```
