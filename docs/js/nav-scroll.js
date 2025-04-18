document.addEventListener('DOMContentLoaded', function() {
    // 获取所有二级标题
    const headers = document.querySelectorAll('.md-content h2');
    
    // 创建一个函数来获取元素的绝对位置
    function getAbsoluteTop(element) {
        let top = 0;
        while (element) {
            top += element.offsetTop;
            element = element.offsetParent;
        }
        return top;
    }

    // 更新导航状态
    function updateNavigation() {
        const scrollPosition = window.scrollY + window.innerHeight * 0.3; // 视窗 30% 处
        
        // 获取所有二级标题的位置
        const headerPositions = Array.from(headers).map(header => {
            return {
                header: header,
                top: getAbsoluteTop(header)
            };
        });
        
        // 找到当前处于活动状态的标题
        let activeHeader = null;
        for (let i = 0; i < headerPositions.length; i++) {
            const current = headerPositions[i];
            const next = headerPositions[i + 1];
            
            // 如果是最后一个标题，使用文档底部作为结束位置
            const bottom = next ? next.top : document.documentElement.scrollHeight;
            
            if (scrollPosition >= current.top && scrollPosition < bottom) {
                activeHeader = current.header;
                break;
            }
        }
        
        // 更新导航状态
        if (activeHeader) {
            const headerText = activeHeader.textContent.trim();
            const navLinks = document.querySelectorAll('.md-nav__link');
            
            navLinks.forEach(link => {
                const linkText = link.textContent.trim();
                const navItem = link.parentElement;
                
                if (linkText === headerText) {
                    // 找到包含子导航的父元素
                    const parentNav = navItem.closest('.md-nav');
                    if (parentNav) {
                        const parentItem = parentNav.parentElement;
                        if (parentItem && parentItem.classList.contains('md-nav__item')) {
                            // 展开父级导航
                            parentItem.classList.add('md-nav__item--active');
                        }
                    }
                    // 展开当前项
                    navItem.classList.add('md-nav__item--active');
                } else if (!headerText.startsWith(linkText)) {
                    // 如果不是当前标题的父级，则折叠
                    navItem.classList.remove('md-nav__item--active');
                }
            });
        }
    }
    
    // 监听滚动事件
    window.addEventListener('scroll', updateNavigation);
    
    // 初始化时检查当前可见的标题
    setTimeout(updateNavigation, 100);
}); 