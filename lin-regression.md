
The linear model in linear regressions

$$
f_{w}(x) = x_{1}w_{1} + \dots + x_{D+1}w_{D+1}+\epsilon = w^Tx+\epsilon
$$


the closed form is: $w^* = (X^TX)^{-1}X^Ty$

How reflog works:


Records *all* movements of `HEAD`, compare to [[log]]. 

Examples: 
- `HEAD@{2}` means "where HEAD used to be two moves ago"
- `master@{one.week.ago}` means "where master used to point to one week ago in this local repo"

Output Example:

```bash
$ git reflog
a1b2c3d HEAD@{0}: reset: moving to HEAD~2
e4f5g6h HEAD@{1}: commit: Further improvements
i8j9k0l HEAD@{2}: commit: Important feature
...
```
