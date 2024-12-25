//全局变量
//是否需要context,正式环境没有的话，需要设置为false;
var NEED_CONTEXT = true;
// 模版调用号,正式运行后，变为常量
var TPL_VERSION = Math.random();// 以后可修改为日期，如20141226...

// 加载时显示的小loading
var LOADING_SMALL = '<span style="color:#4C4F53"><i class="fa fa-2x fa-spin fa-spinner"></i></span>';
// 加载时显示的中loading
var LOADING_MIDDLE = '<span style="color:#4C4F53"><i class="fa fa-3x fa-spin fa-spinner"></i></span>';
// 分页显示数量，默认5
var PAGE_NUM = 5;

$(function() {
	// juicer 默认设置，缓存关闭、转义
	juicer.set({
		'cache' : false
	});
	juicer.set({
		'tag::interpolateOpen' : '!{',
		'tag::noneencodeOpen' : '!!{'
	});

	if ($("#header a")) {
		$("#header a").popover({
			trigger : 'hover'
		});
	}
});

// 获得地址
function getContextPath() {
	var p = document.location.pathname;
	if (NEED_CONTEXT) {
		var index = p.substr(1).indexOf("/");
		p = p.substr(0, index + 1);
		p = window.location.host + p;
	} else {
		p = window.location.host;
	}
	return 'http://' + p;
};

// 显示模态窗口
function showModal(info) {
	var d = dialog(
			{
				id : 'id-modal',
				content : '<span class="font-150"><i class="fa fa-4x fa-spinner fa-spin"></i> &nbsp;&nbsp;&nbsp;'
						+ info + '</span>',
				cancel : false
			}).width(420).height(65);
	d.showModal();
}

// 关闭模态窗口
function hideModal() {
	var d = dialog.get('id-modal');
	d.close().remove();
}

// 显示提示窗口
function showAlert(info) {
	var d = dialog({
		id : 'id-alert',
		title : '消息提示',
		content : info,
		okValue : '确定',
		ok : function() {
		},
		quickClose : true,
		cancel : false
	}).width(320);// 设定宽度
	d.show();
}

// 显示提示窗口 跳页
function showSuccessDial(msg, url) {
	var sfDial = dialog({
		title : '消息提示',
		content : msg,
		okValue : '确定',
		ok : function() {
			if (url == "") {
				window.location.reload(true);
			} else {
				window.location.href = url;
			}
		},
		cancel : false
	}).width(320);// 设定宽度
	sfDial.show();
}

// confirm 对话框
// @by yezhenchi
function showConfirm(info, func) {
	var d = dialog({
		id : 'id-alert',
		title : '消息提示',
		content : info,
		okValue : '确定',
		ok : func,
		quickClose : true,
		cancel : true,
		cancelValue : '取消'
	});
	d.show();
}

// 分页调用方式(pageElement:显示分页的ul容器ID,curPage:当前页号,allPage:总体页数,
// loadDataFunc:回调加载数据函数名称,changePageF:回调改变页号函数名称)
function callPageShow(pageElement, curPage, allPage, loadDataFunc, changePageF) {
	if (allPage < 1) {
		allPage = 1;
	}
	var options = {
		bootstrapMajorVersion : 3,
		size : 'normal',// small, normal, large
		currentPage : curPage,
		numberOfPages : PAGE_NUM,
		totalPages : allPage,
		itemContainerClass : function(type, page, current) {
			return (page === current) ? "active" : "cursor-hand";
		},
		itemTexts : function(type, page, current) {
			switch (type) {
			case "first":
				// return "首 1页";
				return "首页";
			case "prev":
				return "上一页";
			case "next":
				return "下一页";
			case "last":
				// return "尾 " + allPage + "页";
				return "尾页";
			case "page":
				return page;
			}
		},
		tooltipTitles : function(type, page, current) {
			switch (type) {
			case "first":
				return "第一页";
			case "prev":
				return "上一页";
			case "next":
				return "下一页";
			case "last":
				return "最后一页";
			case "page":
				return "第 " + page + " 页";
			}
		},
		onPageClicked : function(e, originalEvent, type, page) {
			changePageF(page);
			loadDataFunc();
		}
	};

	$('#' + pageElement).bootstrapPaginator(options);
}

// tab切换 项目和课程评论 分页显示 ====@by yezhenchi
function callPageShowByClass(pageElement, curPage, allPage, loadDataFunc,
		changePageF, object, val) {
	var options = {
		bootstrapMajorVersion : 3,
		size : 'normal',// small, normal, large
		currentPage : curPage,
		numberOfPages : PAGE_NUM,
		totalPages : allPage,
		itemContainerClass : function(type, page, current) {
			return (page === current) ? "active" : "cursor-hand";
		},
		itemTexts : function(type, page, current) {
			switch (type) {
			case "first":
				// return "首 1页";
				return "首页";
			case "prev":
				return "上一页";
			case "next":
				return "下一页";
			case "last":
				// return "尾 " + allPage + "页";
				return "尾页";
			case "page":
				return page;
			}
		},
		tooltipTitles : function(type, page, current) {
			switch (type) {
			case "first":
				return "第一页";
			case "prev":
				return "上一页";
			case "next":
				return "下一页";
			case "last":
				return "最后一页";
			case "page":
				return "第 " + page + " 页";
			}
		},
		onPageClicked : function(e, originalEvent, type, page) {
			changePageF(page);
			loadDataFunc(val);
		}
	};
	$(object).find('div .' + pageElement).bootstrapPaginator(options);
}

// 文件下载封装方法
function downFile(url) {
	window.location.href = encodeURI(url);
}

// 字符串截取 + ...
function getCutStr(str, len) {
	if (str == null || str == "") {
		return "";
	}
	var returnStr = "";
	var char_length = 0;
	for (var i = 0; i < str.length; i++) {
		var son_str = str.charAt(i);
		encodeURI(son_str).length > 2 ? char_length += 1 : char_length += 0.5;
		if (char_length > len) {
			var sub_len = char_length == len ? i + 1 : i;
			returnStr = str.substr(0, sub_len);
			break;
		}
	}
	if (returnStr == "") {
		returnStr = str;
	} else {
		returnStr = returnStr + "...";
	}
	return returnStr;
}
