<%--使用时： <%@ include file="/WEB-INF/page/common/commonPage.jsp"%>，然后即可调用变量  --%>
<%@ page language="java" pageEncoding="UTF-8"
	contentType="text/html; charset=UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ page import=" com.neusoft.core.util.PCache"%>

<%
	String resourcesUrl = PCache.getHM().get("website.resources.url");
	String contextName = request.getContextPath();
	String resourcesDir = "/resources";
	String websiteVersion = PCache.getHM().get("website.version");
	String websiteName = PCache.getHM().get("website.name");
%>

<%--基础路径，形如：/neumooc，可以使用${base}方式调用 --%>
<c:set var="base" value="<%=contextName%>" />
<%--资源域名/IP地址 和相对地址 --%>
<c:set var="resourcesUrl"
	value="<%=resourcesUrl + contextName + resourcesDir%>" />
<%--版本号字符串，以满足后继更新时，强制更新缓存使用：开发时使用服务器时间戳 --%>
<c:set var="version" value="<%=System.currentTimeMillis()%>" />
<%--网站名称，用于在标题等处显示使用 --%>
<c:set var="websiteName" value="<%=websiteName%>" />

