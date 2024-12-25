package com.neusoft.core.common.pagination;

import javax.servlet.http.HttpServletRequest;

import com.neusoft.core.util.Util;

/**
 * @ClassName: Paginating
 * @Description: 分页信息，存储分页信息和排序信息
 */
public class Paginating {

	/**
	 * @Fields pageSize : 每页显示数据条数
	 */
	private int pageSize;
	/**
	 * @Fields pageIndex : 每页显示结果集数据的起始行号
	 */
	private int pageIndex;
	/**
	 * @Fields sort : 排序的字段名
	 */
	private String sort;
	/**
	 * @Fields order : 排序方式，DESC or ASC
	 */
	private String order;

	/**
	 * 构造方法
	 * 
	 * @param request
	 */
	public Paginating(HttpServletRequest request) {
		pageSize = Util.parseInt(request.getParameter("limit"));
		pageIndex = Util.parseInt(request.getParameter("offset"));
		sort = request.getParameter("sort");
		order = request.getParameter("order");
	}

	/**
	 * 构造方法
	 * 
	 * @param pageSize
	 * @param pageIndex
	 */
	public Paginating(int pageSize, int pageIndex) {
		this.pageSize = pageSize;
		this.pageIndex = pageIndex;
	}

	/**
	 * 构造方法
	 */
	public Paginating() {

	}

	public int getPageSize() {
		return pageSize;
	}

	public void setPageSize(int pageSize) {
		this.pageSize = pageSize;
	}

	public int getPageIndex() {
		return pageIndex;
	}

	public void setPageIndex(int pageIndex) {
		this.pageIndex = pageIndex;
	}

	public String getSort() {
		return sort;
	}

	public void setSort(String sort) {
		this.sort = sort;
	}

	public String getOrder() {
		return order;
	}

	public void setOrder(String order) {
		this.order = order;
	}
}
