package com.neusoft.core.common;

import javax.servlet.http.HttpServletRequest;

import org.springframework.stereotype.Component;

import com.neusoft.core.common.pagination.Paginating;

/**
 * @ClassName: BaseController
 * @Description: 基础Controller类
 */
@Component
public class BaseController extends BaseObject {

	/**
	 * @Title: getPaginating
	 * @Description: 获取分页信息
	 * @param request
	 *            HttpServletRequest
	 * @return Paginating对象
	 */
	protected Paginating getPaginating(HttpServletRequest request) {
		Paginating page = new Paginating(request);
		return page;
	}

	/**
	 * @Title: getPaginating
	 * @Description: 获取分页信息
	 * @param pageSize
	 *            每页显示数据数目
	 * @param pageNo
	 *            页码
	 * @return Paginating
	 */
	protected Paginating getPaginating(int pageSize, int pageNo) {
		int pageIndex = 0;
		if (pageNo > 0) {
			pageIndex = (pageNo - 1) * pageSize;
		}
		Paginating page = new Paginating(pageSize, pageIndex);
		return page;
	}
}
