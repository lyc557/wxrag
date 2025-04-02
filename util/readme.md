### 问题排查
在较新版本的mac上使用命令安装pip install magic-pdf[full] zsh: no matches found: magic-pdf[full]
在 macOS 上，默认的 shell 从 Bash 切换到了 Z shell，而 Z shell 对于某些类型的字符串匹配有特殊的处理逻辑，这可能导致no matches found错误。 可以通过在命令行禁用globbing特性，再尝试运行安装命令
``` bash
setopt no_nomatch
pip install magic-pdf[full]
```

模型文件的路径输入是在"magic-pdf.json"中通过
``` json
{
  "models-dir": "/tmp/models"
}
```

magic-pdf的配置文件位置：
/Users/yangcailu/magic-pdf.json下，
如果你是 mac电脑，支持mps的话，需要把 "device-mode" 的值改为 "mps"