import json
import datetime
import csv
import time
from FacebookAccessToken import AccessToken
from tqdm import tqdm
from sentiment_analysis import sentiment

try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request

# app_id = "<FILL IN>"
# app_secret = "<FILL IN>"  # DO NOT SHARE WITH ANYONE!
file_id = "sugarlandtxgov"
# access_token = app_id + "|" + app_secret
access_token = AccessToken

def request_until_succeed(url):
    req = Request(url)
    success = False
    while success is False:
        try:
            response = urlopen(req)
            if response.getcode() == 200:
                success = True
        except Exception as e:
            print(e)
            time.sleep(5)

            print("Error for URL {}: {}".format(url, datetime.datetime.now()))
            print("Retrying.")

    return response.read()

# Needed to write tricky unicode correctly to csv
def unicode_decode(text):
    try:
        # return text.encode('utf-8').decode()
        return text.encode('cp850', 'replace').decode('cp850')  
    except UnicodeDecodeError:
        return text.encode('utf-8')
        # return text.encode('unicode_escape')


def getFacebookCommentFeedUrl(base_url):

    # Construct the URL string
    # fields = "&fields=id,message,reactions.limit(0).summary(true),created_time,comments,from,attachment"
    fields = "&fields=id,message,reactions.limit(0).summary(true),created_time,comments,from"
    url = base_url + fields

    return url


def getReactionsForComments(base_url):

    reaction_types = ['like', 'love', 'wow', 'haha', 'sad', 'angry']
    reactions_dict = {}   # dict of {status_id: tuple<6>}

    for reaction_type in reaction_types:
        fields = "&fields=reactions.type({}).limit(0).summary(total_count)".format(
            reaction_type.upper())

        url = base_url + fields

        data = json.loads(request_until_succeed(url))['data']

        data_processed = set()  # set() removes rare duplicates in statuses
        for status in data:
            id = status['id']
            count = status['reactions']['summary']['total_count']
            data_processed.add((id, count))

        for id, count in data_processed:
            if id in reactions_dict:
                reactions_dict[id] = reactions_dict[id] + (count,)
            else:
                reactions_dict[id] = (count,)

    return reactions_dict


def processFacebookComment(comment, status_id, parent_id=''):

    # The status is now a Python dictionary, so for top-level items,
    # we can simply call the key.

    # Additionally, some items may not always exist,
    # so must check for existence first
    # print(comment,'\n')
    comment_id = comment['id']

    comment_message = '' if 'message' not in comment or comment['message'] \
        is '' else unicode_decode(comment['message'])
    # Cleaning so it wont affect when saving CSV
    comment_message = comment_message.replace('\n','').replace(',', '').replace(';','')

    try:

        comment_author = unicode_decode(comment['from']['name'])
        comment_author = comment_author.replace('\n','').replace(',', '').replace(';','')
    except:
        comment_author = 'N/A'

    num_reactions = 0 if 'reactions' not in comment else \
        comment['reactions']['summary']['total_count']

    # Time needs special care since a) it's in UTC and
    # b) it's not easy to use in statistical programs.

    comment_published = datetime.datetime.strptime(
        comment['created_time'], '%Y-%m-%dT%H:%M:%S+0000')
    comment_published = comment_published + datetime.timedelta(hours=-5)  # EST
    comment_published = comment_published.strftime(
        '%Y-%m-%d %H:%M:%S')  # best time format for spreadsheet programs

    # Return a tuple of all processed data

    return (comment_id, status_id, parent_id,
            comment_message,
            comment_author,comment_published, str(num_reactions))


def scrapeFacebookPageFeedComments(page_id, access_token):
    with open('{}_facebook_comments.csv'.format(file_id), 'w') as w:
        
        w.write(', '.join(["comment_id", "status_id", "parent_id", "comment_message",
                    "comment_author", "comment_published", "num_reactions",
                    "num_likes", "num_loves", "num_wows", "num_hahas",
                    "num_sads", "num_angrys", "num_special","sentiment", "confidence"]))
        w.write('\n')

        num_processed = 0
        scrape_starttime = datetime.datetime.now()
        after = ''
        base = "https://graph.facebook.com/v3.0"
        parameters = "/?limit={}&access_token={}".format(
            100, access_token)

        print("Scraping {} Comments From Posts: {}\n".format(
            file_id, scrape_starttime))

        with open('{}_facebook_statuses.csv'.format(file_id), 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            # Uncomment below line to scrape comments for a specific status_id
            # reader = [dict(status_id='5550296508_10154352768246509')]

            for status in tqdm(reader):
                has_next_page = True

                while has_next_page:

                    node = "/{}/comments".format(status['status_id'])
                    after = '' if after is '' else "&after={}".format(after)
                    base_url = base + node + parameters + after

                    url = getFacebookCommentFeedUrl(base_url)
                    # print(url)
                    comments = json.loads(request_until_succeed(url))
                    reactions = getReactionsForComments(base_url)

                    for comment in comments['data']:
                        comment_data = processFacebookComment(comment, status['status_id'])
                        reactions_data = reactions[comment_data[0]]

                        # calculate thankful/pride through algebra
                        num_special = int(comment_data[6]) - sum(reactions_data)
                        # w.writerow(comment_data + reactions_data + (num_special, ))
                        w.write(', '.join(comment_data))
                        w.write(', ')
                        reactions_data = str(reactions_data)
                        w.write(reactions_data[1:-1])
                        w.write(', ')
                        w.write(str(num_special))
                        w.write(', ')
                        if len(comment_data[3].split()) > 3:
                            res, conf = sentiment(comment_data[3])
                            w.write('{}, {:.2f}'.format(res, conf))
                        else:
                            w.write(' , ')
                        
                        w.write('\n')

                        if 'comments' in comment:
                            has_next_subpage = True
                            sub_after = ''

                            while has_next_subpage:
                                sub_node = "/{}/comments".format(comment['id'])
                                sub_after = '' if sub_after is '' else "&after={}".format(
                                    sub_after)
                                sub_base_url = base + sub_node + parameters + sub_after

                                sub_url = getFacebookCommentFeedUrl(
                                    sub_base_url)
                                sub_comments = json.loads(
                                    request_until_succeed(sub_url))
                                sub_reactions = getReactionsForComments(
                                    sub_base_url)

                                for sub_comment in sub_comments['data']:
                                    sub_comment_data = processFacebookComment(
                                        sub_comment, status['status_id'], comment['id'])
                                    sub_reactions_data = sub_reactions[
                                        sub_comment_data[0]]

                                    num_sub_special = int(sub_comment_data[6]) - sum(sub_reactions_data)

                                    # w.writerow(sub_comment_data +
                                    #            sub_reactions_data + (num_sub_special,))

                                    w.write(', '.join(sub_comment_data))
                                    w.write(', ')
                                    sub_reactions_data = str(sub_reactions_data)
                                    w.write(sub_reactions_data[1:-1])
                                    w.write(', ')
                                    w.write(str(num_sub_special))
                                    w.write(', ')
                                    if len(sub_comment_data[3].split()) > 3:
                                        sub_res, sub_conf = sentiment(sub_comment_data[3])
                                        w.write('{}, {:.2f}'.format(sub_res, sub_conf))
                                    else:
                                        w.write(' , ')
                                    w.write('\n')

                                    num_processed += 1
                                    if num_processed % 100 == 0:
                                        print("{} Comments Processed: {}".format(
                                            num_processed,
                                            datetime.datetime.now()))

                                if 'paging' in sub_comments:
                                    if 'next' in sub_comments['paging']:
                                        sub_after = sub_comments[
                                            'paging']['cursors']['after']
                                    else:
                                        has_next_subpage = False
                                else:
                                    has_next_subpage = False

                        # output progress occasionally to make sure code is not
                        # stalling
                        num_processed += 1
                        # if num_processed % 100 == 0:
                        #     print("{} Comments Processed: {}".format(
                        #         num_processed, datetime.datetime.now()))

                    if 'paging' in comments:
                        if 'next' in comments['paging']:
                            after = comments['paging']['cursors']['after']
                        else:
                            has_next_page = False
                    else:
                        has_next_page = False

        print("\nDone!\n{} Comments Processed in {}\n".format(
            num_processed, datetime.datetime.now() - scrape_starttime))


if __name__ == '__main__':
    scrapeFacebookPageFeedComments(file_id, access_token)
