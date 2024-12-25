<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ include file="/WEB-INF/page/common/commonPage.jsp"%>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="utf-8" />
<title>500-${websiteName}</title>
</head>
<body>
	<!-- 页头开始 -->
	<%@ include file="/WEB-INF/page/common/header.jsp"%>
	<%@ include file="/WEB-INF/page/common/headerbar.jsp"%>
	<!-- 页头结束 -->

	<!-- 左侧菜单导航开始 -->
	<%@ include file="/WEB-INF/page/common/sidebar.jsp"%>
	<!-- 左侧菜单导航结束 -->

	<div id="main" role="main">
		<div id="ribbon">
			<ol class="breadcrumb">
				<li><a href="${base}/home/index">首页</a></li>
				<li>500</li>
			</ol>
		</div>
		<div id="content">
			<div class="row">
				<div class="col-xs-12 col-sm-12 col-md-12 col-lg-12">
					<div class="row">
						<div class="text-center error-box">
							<div class="error-text-2 bounceInDown animated">
								<img src="${resourcesUrl}/common/img/error_500_bg.png">
							</div>
							<div class="padding-10">
								<a href="javascript:location.href='${base}/home/index'"><img
									src="${resourcesUrl}/common/img/bt_index.png"> </a> <a
									href="javascript:history.go(-1)"><img
									src="${resourcesUrl}/common/img/bt_back.png"> </a>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>

	<!-- 包含footer，引入公用 JS -->
	<%@ include file="/WEB-INF/page/common/footer.jsp"%>

</body>
</html>
