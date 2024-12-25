<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ include file="/WEB-INF/page/common/commonPage.jsp"%>
<%@ taglib prefix="sec"
	uri="http://www.springframework.org/security/tags"%>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="utf-8" />
<title>首页-${websiteName}</title>
</head>
<body>

	<!-- 页头开始 -->
	<%@ include file="/WEB-INF/page/common/header.jsp"%>
	<%@ include file="/WEB-INF/page/common/headerbar.jsp"%>
	<!-- 页头结束 -->

	<!-- 左侧菜单导航开始 -->
	<%@ include file="/WEB-INF/page/common/sidebar.jsp"%>
	<!-- 左侧菜单导航结束 -->

	<!-- MAIN PANEL -->
	<div id="main" role="main">
		<!-- 路径导航开始 -->
		<div id="ribbon">
			<ol class="breadcrumb">
				<li>首页</li>
			</ol>
		</div>
		<!-- 路径导航结束 -->
		<div id="content">
			<h4>欢迎登录系统。</h4>
		</div>
	</div>

	<%@ include file="/WEB-INF/page/common/footer.jsp"%>
</body>
</html>