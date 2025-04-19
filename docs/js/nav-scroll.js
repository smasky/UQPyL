document.addEventListener('DOMContentLoaded', function() {
    // 检测是否为移动设备
    const isMobile = window.innerWidth <= 600;
    
    // 获取所有二级标题
    const headers = document.querySelectorAll('.md-content h2');
    
    // 如果是移动设备且没有足够的二级标题，不执行滚动导航
    if (isMobile && headers.length < 2) {
        return;
    }
    
    // 创建一个函数来获取元素的绝对位置
    function getAbsoluteTop(element) {
        let top = 0;
        while (element) {
            top += element.offsetTop;
            element = element.offsetParent;
        }
        return top;
    }

    // 节流函数，限制事件触发频率
    function throttle(func, delay) {
        let lastCall = 0;
        return function(...args) {
            const now = new Date().getTime();
            if (now - lastCall >= delay) {
                lastCall = now;
                return func.apply(this, args);
            }
        };
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
    
    // 使用节流函数优化滚动事件
    const throttledUpdate = throttle(updateNavigation, 100);
    
    // 监听滚动事件
    window.addEventListener('scroll', throttledUpdate);
    
    // 监听窗口大小变化事件
    window.addEventListener('resize', function() {
        // 重新检测是否为移动设备
        const newIsMobile = window.innerWidth <= 600;
        if (newIsMobile !== isMobile) {
            // 强制刷新页面以重置导航状态
            window.location.reload();
        }
    });
    
    // 初始化时检查当前可见的标题
    setTimeout(updateNavigation, 100);
}); 