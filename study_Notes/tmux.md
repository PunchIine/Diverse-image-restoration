# tmux一些常用命令记录

## 创建session

```
tmux new -s session_name
```

在session中**创建新窗口**： ctrl b + c

## 在窗口间切换

ctrl b + 窗口号

## 离开（detach）当前窗口

ctrl b + d

## 查看所有session及窗口

```
tmux ls
```

## 进入指定session

```
tmux a -t session_name
```

## kill指定session

```
tmux kill-session -t session_name
```

## 在session内切换至另一个session

```
tmux switch -t session_name
```

![](/home/lazy/.config/marktext/images/2022-07-23-15-28-09-image.png)
