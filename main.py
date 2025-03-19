# directed by STAssn
# 运行main.py查看效果

import sys
from ui_set import QApplication, MyWidget

app = QApplication(sys.argv)
window = MyWidget()
window.show()
sys.exit(app.exec_())
