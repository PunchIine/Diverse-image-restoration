快速排序是一种**分治算法**，分治算法一般分为三步：

1. 分成几个子问题
2. 递归处理子问题
3. 子问题合并

**模板：**



```c++
void quick_sort(int q[], int l, int r)
{
    //递归的终止情况
    if(l >= r) return;
    //第一步：分成子问题
    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while(i < j)
    {
        do i++; while(q[i] < x);
        do j--; while(q[j] > x);
        if(i < j) swap(q[i], q[j]);
    }
    //第二步：递归处理子问题
    quick_sort(q, l, j), quick_sort(q, j + 1, r);
    //第三步：子问题合并.快排这一步不需要操作，但归并排序的核心在这一步骤
}
```



**快排思路概述**：

1. 有一数组$q$，现对其排序，数组左端点为$l$，右端点为$r$

2. 确定划分边界$x$

3. 将$q$分为$>=x$和$<=x$的两个子数组，下面记子数组为$p$

4. 定义两个指针$i$，$j$，分别以该子数组的$l$和$r$为起点相向进行遍历，分别在$p[i]>x$和$p[j]<x$的值后停下，二者都停下后交换$p[i]$与$p[j]$的位置

    + $i$的含义：该子数组中，在$i$之前的所有元素都$<=x$, 即$q[l, i-1]<=x$

    + $j$的含义：该子数组中，在$j$之后的所有元素都$>=x$, 即$q[j+1, r]>=x$

    + 即二指针停下时满足$p[i]>x$，$p[j]<x$，此时我们交换二者，如此循环至$i>=j$

    + 注意最后一次循环中if语句不会被执行，则在最后一次循环中[. . .j x x i . . .]   ,  [. . . i(j) . . . ] ...

        此时只能保证$i >= j$和$q[l..i-1] <= x$，$ q[i] >= x$和$q[j+1..r] >= x$，$q[j] <= x$，由$q[l..i-1] <=x$，$i >= j(i-1 >= j-1)$ 和 $q[j] <= x$ 可以得到 $q[l..j] <= x$，又因为$q[j+1..r] >= x$，所以，$q[l..j] <= x,q[j+1..r] >= x$

5. 由此我们以$j$作划分，递归处理子问题

6. 注意$q[i]<x$，$q[j]>x$切忌改为$<=，>=$否则会遇到边界问题，这里不细说



代码实现：

c++

```c++
#include <iostream>

using namespace std;

const int N = 1000010;

int q[N];

void q_sort(int l, int r)
{
        if(l == r) return;
        
        int x = q[(l+r) >> 1];
        int i = l-1, j = r+1;
        while(i < j)
        {
            do(i++); while(q[i]<x);
            do(j--); while(q[j]>x);
            if(i < j) swap(q[i], q[j]);
        }
        q_sort(l, j);
        q_sort(j+1, r);
}

int main()
{
    int n;
    cin >> n;
    for(int i = 0; i < n; i++) scanf("%d", &q[i]);
    q_sort(0, n-1);
    for(int i = 0; i < n; i++) printf("%d ", q[i]);
    return 0;
}
```

python

它真的好优雅，我哭死

```python
n =int(input())
nums =list(map(int, input().split()))

def quick_sort(nums):

    if(len(nums) <=1): return nums;

    privot =nums[len(nums)//2]
    left =[x for x in nums if x < privot]
    mid =[x for x in nums if x == privot]
    right =[x for x in nums if x> privot]
    return quick_sort(left) +mid + quick_sort(right)


if __name__ =="__main__":
    nums =quick_sort(nums)
    print(" ".join(list(map(str, nums))))

```

