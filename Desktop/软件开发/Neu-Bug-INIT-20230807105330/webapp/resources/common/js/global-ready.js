/**
 * 这里是平台全局JS方法区
 * 各种需要页面加载后执行的公用内容均放在此处，统一调用
 */
 
$(document).ready(function () {
	var overShowHead = $("#overShowHead");
	if(overShowHead.html()) {
		overShowHead.hover(function(){
			overShowHead.addClass('open');
		}, function(){
			overShowHead.removeClass('open');
		});
	}
});
