/**
 * 登录js
 */
// init
$(function() {

	// 初始化
	initForm();

	// 情况提示信息
	$("input").focus(function() {
		$("#tipInfo").html("");
	});
});

/**
 * 初始化form
 */
function initForm() {
	// 设置from
	setFrom();

	// 登录
	$('#loginBtn').click(function() {
		var url = getContextPath() + "/login/checkLogin";
		$("#loginForm").attr("action", url);
		$("#loginForm").submit();
	});
}

// 设置form
function setFrom() {
	// 提示信息
	var emsg = "<i class='fa fa-exclamation-triangle'></i> ";

	// 校验form
	var vform = $("#loginForm").validate({
		rules : {
			userName : {
				required : true
			},
			password : {
				required : true
			}
		},
		messages : {
			userName : {
				required : '请输入用户名'
			},
			password : {
				required : '请输入密码'
			}
		},
		submitHandler : function(form) {
			$(form).ajaxSubmit({
				success : function(data) {
					if ("success" == data.RET_CODE) {
						var userRequestUrl = $("#userRequestUrl").val();
						if (userRequestUrl != null && userRequestUrl != "") {
							var vurl = userRequestUrl;
							location.replace(vurl);
						} else {
							var vurl = "/home/index";
							location.replace(getContextPath() + vurl);
						}
					} else {
						$("#tipInfo").addClass("tip-error-info");
						$("#tipInfo").html(emsg + "用户名或密码错误");
					}
				}
			});
		},
		errorPlacement : function(error, element) {
			error.insertAfter(element.parent());
			element[0].focus();
			return false;
		}
	});
}