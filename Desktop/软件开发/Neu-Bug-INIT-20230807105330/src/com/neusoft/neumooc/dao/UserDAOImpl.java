package com.neusoft.neumooc.dao;

import org.springframework.stereotype.Repository;

import com.neusoft.core.common.BaseDAO;
import com.neusoft.core.common.exception.DAOException;
import com.neusoft.neumooc.vo.UserVO;

/**
 * @ClassName: UserDAOImpl
 * @Description: 用户管理数据操作类
 * @date 2016年7月13日
 */
@Repository
public class UserDAOImpl extends BaseDAO implements UserDAO {

	// 用户信息查询 根据用户名
	private static final String SQL_GET_USER_BY_USER_NAME = "SELECT "
			+ " u.userId, u.userName, u.password, u.roleId, u.realName, u.email, "
			+ " (SELECT COUNT(p.prjid) FROM bug_projects p WHERE p.userid=u.userid ) prjCount "
			+ " FROM bug_users u WHERE u.userName = ?  AND u.status=1  ";

	/**
	 * @Title: getUserByUserName
	 * @Description: 用户信息查询 根据用户名
	 * @param userName
	 *            用户名
	 * @return UserVO 用户信息
	 * @throws DAOException
	 */
	@Override
	public UserVO getUserByUserName(String userName) throws DAOException {
		UserVO vo = null;
		try {
			Object[] args = { userName };
			vo = this.queryForBean(SQL_GET_USER_BY_USER_NAME, UserVO.class,
					args);
		} catch (Exception e) {
			throw new DAOException(
					"class UserDAOImpl-->method getUserByUserName error", e);
		}
		return vo;
	}

}
