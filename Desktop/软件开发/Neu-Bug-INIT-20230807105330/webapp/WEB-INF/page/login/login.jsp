<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<%@ include file="/WEB-INF/page/common/commonPage.jsp"%>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<!-- Head -->
<head>
<meta charset="utf-8" />
<title></title>
<%@ include file="/WEB-INF/page/common/header.jsp"%>
<!--Page Related styles-->
</head>
<!-- /Head -->
<!-- Body -->
<body>
	<header id="header"
		style="background: #f4f4f4 !important; height: 80px;">
		<div id="logo-group">
			<span id="logo"> <img style="width: 700px; height: 48px;"
				src="${resourcesUrl}/img/logo.png" alt="">
			</span>
		</div>
	</header>
	<div id="content" class="container">
		<div class="row" style="margin-top: 65px;">
			<div class="col-md-7 col-lg-8 hidden-xs hidden-sm">
				<h1 class="txt-color-red login-header-big">&nbsp;</h1>
				<div style="margin-top: 24px; margin-bottom: 20px">
					<img style="width: 100%; height: 325px;" alt=""
						class="display-image" src="${resourcesUrl}/img/login.jpg">
				</div>
			</div>
			<div class="col-xs-12 col-sm-12 col-md-5 col-lg-4">
				<div class="well no-padding">
					<form id="loginForm" method="post" class="smart-form client-form"
						AutoComplete="off">
						<header class="page-title txt-color-blueDark hidden-tablet">
							<strong>系统登录</strong>
						</header>
						<fieldset>
							<input type="hidden" id="userRequestUrl" name="userRequestUrl"
								value="${userRequestUrl}" />
							<div class="row" style="height: 30px;"></div>
							<div class="row">
								<section
									class="col col-lg-2 col-md-3 col-sm-3 hidden-xs no-padding">
									<label class="label text-right">用户名</label>
								</section>
								<section class="col col-lg-10 col-md-9 col-sm-9 col-xs-12">
									<label class="input"> <i class="icon-append fa fa-user"></i>
										<input type="text" name="userName" maxlength="30"> <b
										class="tooltip tooltip-top-right"><i
											class="fa fa-user txt-color-teal"></i> 请输入用户名</b></label>
								</section>
							</div>
							<div class="row">
								<section
									class="col col-lg-2 col-md-3 col-sm-3 hidden-xs no-padding">
									<label class="label text-right">密码</label>
								</section>
								<section class="col col-lg-10 col-md-9 col-sm-9 col-xs-12">
									<label class="input"> <i class="icon-append fa fa-lock"></i>
										<input id="password" type="password" name="password"
										maxlength="20"> <b class="tooltip tooltip-top-right"><i
											class="fa fa-lock txt-color-teal"></i> 请输入密码</b>
									</label>
								</section>
							</div>
							<div class="row">
								<section
									class="col col-lg-2 col-md-3 col-sm-3 hidden-xs no-padding">
								</section>
								<section class="col col-lg-10 col-md-9 col-sm-9 col-xs-12">
									<div id="tipInfo"></div>
								</section>
							</div>
							<div class="row" style="height: 35px;"></div>
						</fieldset>
						<footer>
							<button type="button" class="btn btn-primary" id="loginBtn">登录</button>
						</footer>
					</form>
				</div>
			</div>
		</div>
	</div>

	<%@ include file="/WEB-INF/page/common/footer.jsp"%>

	<!-- 登录js -->
	<script src="${resourcesUrl}/js/login/login.js"></script>
</body>
</html>